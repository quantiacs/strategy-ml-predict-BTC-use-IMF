# Machine Learning - predict futures BTC use IMF Commodity Data

This trading strategy is designed for the [Quantiacs](https://quantiacs.com/contest) platform, which hosts competitions
for trading algorithms. Detailed information about the competitions is available on
the [official Quantiacs website](https://quantiacs.com/contest).

## How to Run the Strategy

### In an Online Environment

The strategy can be executed in an online environment using Jupiter or JupiterLab on
the [Quantiacs personal dashboard](https://quantiacs.com/personalpage/homepage). To do this, clone the template in your
personal account.

### In a Local Environment

To run the strategy locally, you need to install the [Quantiacs Toolbox](https://github.com/quantiacs/toolbox).

## Strategy Overview

This Jupyter notebook outlines a machine learning strategy for predicting Bitcoin futures using both cryptofutures data
and IMF Commodity Data. The strategy employs a RidgeClassifier model, trained with the top 5 features selected by
Recursive Feature Elimination (RFE) from the sklearn.feature_selection library. For the cryptofutures data, features
include a trend indicator, the stochastic oscillator, volatility, and volume. From the IMF Commodity Data, the features
used are simple moving averages (SMA) and logarithmic values. The strategy includes comprehensive code for data loading,
preprocessing, model training, and prediction generation. It utilizes libraries such as pandas, xarray, numpy, and
sklearn for data manipulation, statistical analysis, and machine learning tasks. The notebook is structured to guide
through the process of training a model to predict future price movements of Bitcoin futures, based on historical data
and commodity price trends, and outlines how to backtest the strategy using predefined metrics and intervals.

**Strategy idea**: We will open crypto futures BTC positions as predicted by the RidgeClassifier.

**Features for learning**:

5 best features use RFE (from sklearn.feature_selection import RFE)

Cryptofutures data:

* a trend indicator;
* the stochastic oscillator;
* volatility;
* volume.

IMF Commodity Data (monthly data)

* sma
* log

```python
import logging

import pandas as pd
import xarray as xr
import numpy as np

import qnt.data as qndata  # load and manipulate data
import qnt.backtester as qnbt  # backtester
import qnt.stats as qnstats  # statistical functions for analysis
import qnt.ta as qnta  # indicators library


def load_data(period):
    def align_data_by_time(data, data_for_align):
        data_for_outer = xr.align(data.time, data_for_align, join='outer')[1]
        ff = data_for_outer.ffill(dim='time')
        r = ff.sel(time=data.time)
        return r

    crypto_futures = qndata.cryptofutures_load_data(tail=period)
    commodity = align_data_by_time(
        data=crypto_futures,
        data_for_align=qndata.imf_load_commodity_data(tail=period))

    return dict(commodity=commodity,
                crypto_futures=crypto_futures), crypto_futures.time.values


def window(data, max_date: np.datetime64, lookback_period: int):
    min_date = max_date - np.timedelta64(lookback_period, 'D')
    return dict(
        crypto_futures=data['crypto_futures'].copy(True).sel(time=slice(min_date, max_date)),
        commodity=data['commodity'].copy(True).sel(time=slice(min_date, max_date))
    )


def create_model():
    from sklearn.linear_model import RidgeClassifier
    from sklearn.feature_selection import RFE
    model = RidgeClassifier(random_state=18)
    count_best_features = 5
    rfe = RFE(model, n_features_to_select=count_best_features)
    return rfe


def get_features(futures_commodity):
    data = futures_commodity['crypto_futures']

    trend = qnta.roc(qnta.lwma(data.sel(field='close'), 70), 1)

    # stochastic oscillator:
    k, d = qnta.stochastic(data.sel(field='high'), data.sel(field='low'), data.sel(field='close'), 14)

    volatility = qnta.tr(data.sel(field='high'), data.sel(field='low'), data.sel(field='close'))
    volatility = volatility / data.sel(field='close')
    volatility = qnta.lwma(volatility, 14)

    volume = data.sel(field='vol')
    volume = qnta.sma(volume, 5) / qnta.sma(volume, 60)
    volume = volume.where(np.isfinite(volume), 0)

    crypto_features = xr.concat(
        [trend, d, volatility, volume],
        pd.Index(
            ['trend', 'stochastic_d', 'volatility', 'volume'],
            name='field'
        )
    )

    data_commodity = futures_commodity['commodity']
    data_commodity = data_commodity.rename({'asset': 'field'})

    sma = qnta.sma(data_commodity, 30) / qnta.sma(data_commodity, 60)
    log = np.log(data_commodity)

    commodity_features = xr.concat(
        [sma, log],
        dim='field'
    )

    commodity_merge = commodity_features.sel(time=crypto_features.time)
    features = xr.concat([crypto_features, commodity_merge], dim='field')
    return features.transpose('time', 'field', 'asset')


def get_target_classes(futures_commodity):
    """Builds target classes which will be later predicted."""
    data = futures_commodity['crypto_futures']

    price_current = data.sel(field='close')
    price_future = qnta.shift(price_current, -1)

    class_positive = 1
    class_negative = 0

    target_is_price_up = xr.where(price_future > price_current, class_positive, class_negative)

    return target_is_price_up


def create_and_train_models(futures_commodity):
    """Create and train the models working on an asset-by-asset basis."""

    features_all = get_features(futures_commodity)
    target_all = get_target_classes(futures_commodity)

    models = dict()
    asset_name_all = futures_commodity['crypto_futures'].coords['asset'].values
    for asset_name in asset_name_all:

        # drop missing values:
        target_for_asset = target_all.sel(asset=asset_name).dropna('time', 'any')
        features_for_asset = features_all.sel(asset=asset_name).dropna('time', 'any')

        # align features and targets:
        target_for_learn_df, feature_for_learn_df = xr.align(target_for_asset,
                                                             features_for_asset,
                                                             join='inner')

        is_few_data_for_train = len(target_for_learn_df.time) < 10
        if is_few_data_for_train:
            continue

        model = create_model()

        try:
            model.fit(feature_for_learn_df.values, target_for_learn_df)
            models[asset_name] = model
        except KeyboardInterrupt as e:
            raise e
        except:
            logging.exception('model training failed')

    return models


def predict(models, futures_commodity):
    """Performs prediction and generates output weights.
       Generation is performed for several days in order to speed
       up the evaluation.
    """
    data = futures_commodity['crypto_futures']
    weights = xr.zeros_like(data.sel(field='close'))

    asset_name_all = data.coords['asset'].values
    for asset_name in asset_name_all:
        if asset_name in models:
            model = models[asset_name]
            features_all = get_features(futures_commodity)
            features_cur = features_all.sel(asset=asset_name).dropna('time', 'any')
            if len(features_cur.time) < 1:
                continue
            try:
                weights.loc[dict(asset=asset_name, time=features_cur.time.values)] = model.predict(
                    features_cur.values)
            except KeyboardInterrupt as e:
                raise e
            except:
                logging.exception('model prediction failed')

    return weights


weights = qnbt.backtest_ml(
    train=create_and_train_models,
    predict=predict,
    train_period=10 * 365,  # the data length for training in calendar days
    retrain_interval=365,  # how often we have to retrain models (calendar days)
    retrain_interval_after_submit=1,  # how often retrain models after submission during evaluation (calendar days)
    predict_each_day=False,  # Is it necessary to call prediction for every day during backtesting?
    # Set it to true if you suspect that get_features is looking forward.
    competition_type='cryptofutures',  # competition type
    lookback_period=365,  # how many calendar days are needed by the predict function to generate the output
    start_date='2014-01-01',  # backtest start date
    build_plots=True,  # do you need the chart?
    load_data=load_data,
    window=window,
)
```
