"""
This algorithm demonstrates the concept of long-short equity. It uses a
combination of factors to construct a ranking of securities in a liquid
tradable universe. It then goes long on the highest-ranked securities and short
on the lowest-ranked securities.

For information on long-short equity strategies, please see the corresponding
lecture on our lectures page:

https://www.quantopian.com/lectures

This algorithm was developed as part of Quantopian's Lecture Series. Please
direct and questions, feedback, or corrections to feedback@quantopian.com
"""

import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline, CustomFactor
import numpy

import quantopian.pipeline.data.factset.estimates as fe

from quantopian.pipeline.factors import SimpleMovingAverage, Returns

from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline

from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.data.builtin import USEquityPricing

# Constraint Parameters
MAX_GROSS_LEVERAGE = 1.0
TOTAL_POSITIONS = 600

# Here we define the maximum position size that can be held for any
# given stock. If you have a different idea of what these maximum
# sizes should be, feel free to change them. Keep in mind that the
# optimizer needs some leeway in order to operate. Namely, if your
# maximum is too small, the optimizer may be overly-constrained.
MAX_SHORT_POSITION_SIZE = 1.0 / 100
MAX_LONG_POSITION_SIZE = 1.0 / 1000


def initialize(context):
    """
    A core function called automatically once at the beginning of a backtest.

    Use this function for initializing state or other bookkeeping.

    Parameters
    ----------
    context : AlgorithmContext
        An object that can be used to store state that you want to maintain in
        your algorithm. context is automatically passed to initialize,
        before_trading_start, handle_data, and any functions run via schedule_function.
        context provides the portfolio attribute, which can be used to retrieve information
        about current positions.
    """

    algo.attach_pipeline(make_pipeline(context), 'long_short_equity_template')

    # Attach the pipeline for the risk model factors that we
    # want to neutralize in the optimization step. The 'risk_factors' string is
    # used to retrieve the output of the pipeline in before_trading_start below.
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_factors')

    # Schedule our rebalance function
    algo.schedule_function(func=rebalance,
                           date_rule=algo.date_rules.week_start(),
                           time_rule=algo.time_rules.market_open(hours=0, minutes=30),
                           half_days=True)

    # Record our portfolio variables at the end of day
    algo.schedule_function(func=record_vars,
                           date_rule=algo.date_rules.every_day(),
                           time_rule=algo.time_rules.market_close(),
                           half_days=True)


class Momentum(CustomFactor):
    # Default inputs
    inputs = [USEquityPricing.close]

    # Compute momentum
    def compute(self, today, assets, out, close):
        out[:] = close[-1] / close[0]


class Factor(CustomFactor):
    """
    EPS UPGRADES / DOWNGRADES
    """
    inputs = [fe.PeriodicConsensus.slice('EPS', 'qf', 1).up, fe.PeriodicConsensus.slice('EPS', 'qf', 1).down]
    window_length = 1
    window_safe = True

    def compute(self, today, assets, out, up, down):
        out[:] = 1.0 * up[-1] / (up[-1] + down[-1])


def create_factor():
    # Base universe set to the QTradableStocksUS
    qtu = QTradableStocksUS();

    fq1_eps_cons = fe.PeriodicConsensus.slice('EPS', 'qf', 1);

    fq1_percent = Factor();
    fq1_mean = fq1_eps_cons.mean.latest.abs();

    fq1_up = fe.PeriodicConsensus.slice('EPS', 'qf', 1).up.latest;
    fq1_down = fe.PeriodicConsensus.slice('EPS', 'qf', 1).down.latest;

    fq1_TotalEst = fq1_up + fq1_down

    fq1_stdv = fq1_eps_cons.std_dev.latest;
    # abs(close[i-9])
    fq1_VarCoef = fq1_stdv / fq1_mean;

    alpha_factor = fq1_percent / fq1_VarCoef;
    # alpha_factor = fq1_eps_cons_up - fq1_eps_cons_down

    return alpha_factor, fq1_TotalEst, qtu


class StdDev(CustomFactor):
    def compute(self, today, asset_ids, out, values):
        # Calculates the column-wise standard deviation, ignoring NaNs
        out[:] = numpy.nanstd(values, axis=0)


def make_pipeline(context):
    """
    A function that creates and returns our pipeline.

    We break this piece of logic out into its own function to make it easier to
    test and modify in isolation. In particular, this function can be
    copy/pasted into research and run by itself.

    Returns
    -------
    pipe : Pipeline
        Represents computation we would like to perform on the assets that make
        it through the pipeline screen.
    """

    alpha_factor, fq1_TotalEst, qtu = create_factor()

    screenAlpha = qtu & ~alpha_factor.isnull() & alpha_factor.isfinite() & (fq1_TotalEst > 4)

    # Winsorize to remove extreme outliers
    alpha_winsorized = alpha_factor.winsorize(min_percentile=0.01,
                                              max_percentile=0.99,
                                              mask=screenAlpha)

    # Zscore and rank to get long and short (positive and negative) alphas to use as weights
    alpha_rank = alpha_winsorized.rank().zscore()

    ten_day_momentum = Momentum(window_length=10)
    std_dev = StdDev(
        inputs=[USEquityPricing.close],
        window_length=5
    )
    value = Fundamentals.ebit.latest / Fundamentals.enterprise_value.latest
    quality = Fundamentals.roe.latest
    sentiment_score = SimpleMovingAverage(
        inputs=[stocktwits.bull_minus_bear],
        window_length=3,
    )

    universe = QTradableStocksUS()

    # We winsorize our factor values in order to lessen the impact of outliers
    # For more information on winsorization, please see
    # https://en.wikipedia.org/wiki/Winsorizing
    std_dev_winsorized = std_dev.winsorize(min_percentile=0.05, max_percentile=0.95)
    ten_day_momentum_winsorized = ten_day_momentum.winsorize(min_percentile=0.05, max_percentile=0.95)
    value_winsorized = value.winsorize(min_percentile=0.05, max_percentile=0.95)
    quality_winsorized = quality.winsorize(min_percentile=0.05, max_percentile=0.95)
    sentiment_score_winsorized = sentiment_score.winsorize(
        min_percentile=0.05,
        max_percentile=0.95
    )

    # Here we combine our winsorized factors, z-scoring them to equalize their influence
    combined_factor = (
            2 * alpha_rank -
            std_dev_winsorized.zscore() / 2 +
            ten_day_momentum_winsorized.zscore() / 2 +
            value_winsorized.zscore() +
            quality_winsorized.zscore() +
            sentiment_score_winsorized.zscore()
    )

    # Build Filters representing the top and bottom baskets of stocks by our
    # combined ranking system. We'll use these as our tradeable universe each
    # day.
    longs = combined_factor.top(TOTAL_POSITIONS // 2, mask=universe)
    shorts = combined_factor.bottom(TOTAL_POSITIONS // 2, mask=universe)

    # The final output of our pipeline should only include
    # the top/bottom 300 stocks by our criteria
    long_short_screen = (longs | shorts)

    # Create pipeline
    pipe = Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts,
            'combined_factor': combined_factor
        },
        screen=long_short_screen
    )
    return pipe


def before_trading_start(context, data):
    """
    Optional core function called automatically before the open of each market day.

    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        An object that provides methods to get price and volume data, check
        whether a security exists, and check the last time a security traded.
    """
    # Call algo.pipeline_output to get the output
    # Note: this is a dataframe where the index is the SIDs for all
    # securities to pass my screen and the columns are the factors
    # added to the pipeline object above
    context.pipeline_data = algo.pipeline_output('long_short_equity_template')

    # This dataframe will contain all of our risk loadings
    context.risk_loadings = algo.pipeline_output('risk_factors')


def record_vars(context, data):
    """
    A function scheduled to run every day at market close in order to record
    strategy information.

    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        See description above.
    """
    # Plot the number of positions over time.
    algo.record(num_positions=len(context.portfolio.positions))


# Called at the start of every month in order to rebalance
# the longs and shorts lists
def rebalance(context, data):
    """
    A function scheduled to run once every Monday at 10AM ET in order to
    rebalance the longs and shorts lists.

    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        See description above.
    """
    # Retrieve pipeline output
    pipeline_data = context.pipeline_data

    risk_loadings = context.risk_loadings

    # Here we define our objective for the Optimize API. We have
    # selected MaximizeAlpha because we believe our combined factor
    # ranking to be proportional to expected returns. This routine
    # will optimize the expected return of our algorithm, going
    # long on the highest expected return and short on the lowest.
    objective = opt.MaximizeAlpha(pipeline_data.combined_factor)

    # Define the list of constraints
    constraints = []
    # Constrain our maximum gross leverage
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_LEVERAGE))

    # Require our algorithm to remain dollar neutral
    constraints.append(opt.DollarNeutral())

    # Add the RiskModelExposure constraint to make use of the
    # default risk model constraints
    neutralize_risk_factors = opt.experimental.RiskModelExposure(
        risk_model_loadings=risk_loadings,
        version=0
    )
    constraints.append(neutralize_risk_factors)

    # With this constraint we enforce that no position can make up
    # greater than MAX_SHORT_POSITION_SIZE on the short side and
    # no greater than MAX_LONG_POSITION_SIZE on the long side. This
    # ensures that we do not overly concentrate our portfolio in
    # one security or a small subset of securities.
    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))

    # Put together all the pieces we defined above by passing
    # them into the algo.order_optimal_portfolio function. This handles
    # all of our ordering logic, assigning appropriate weights
    # to the securities in our universe to maximize our alpha with
    # respect to the given constraints.
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )