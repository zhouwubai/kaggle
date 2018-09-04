# how to preprocess

So for preprocessing, we do following simple approach
* remove rows with nan in columns `actual_delivery_time`, `estimated_store_to_consumer_driving_duration`
* consider categorical nan as a new category in `market_id`, `order_protocol`, `store_primary_category`, 'store_id'
  and calculate one-hot coding
* in columns `total_onshift_dashers`, `total_busy_dashers`, `total_outstanding_orders`, I realized they are all missed in
  same rows. we can use simple imputation approach to handle them.

# what model to use

Usually in industry, we try to leverage on boosting trees or random forest for robust performance.
RandomForest will be a better choice in most cases, since it is good to handle missing values and
usually has good generalization ability, i.e., less variance, becasue every tree (weak learner is independent).
Moreover it is easier to parallelize and easy to tune.

The advantage of boosting tree is more flexible
for different objective functions so can be used in more complex tasks. but since it uses boosting techniques, i.e.,
tree are not independent, bias might be low, but variance will be high. It also more sensitive to noisy data.

From business perspective, for ETA, sometimes 1 mins (60s) error should be fine, and user will get used to it,
but if somehow it get large error several time, users will complain.

# how to evaluate and check insights
* cross validation on RSME
* check feature importance to find insights on different features

# New features
* instead of some absolute values, use statistical numbers, such as ratio between different dasher, ration of
avaliable dasher to outstanding orders, and even variance of some feature.

# how to handle online senarios (testing)?
* train the model offline and save as file
* load the model
* process the input that will be a dictinary for each instance instead of whole dataframe. File `process` gives an idea about
 the structure.

# how to automate unit test ?
* pytest