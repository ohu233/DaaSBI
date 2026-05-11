-- Active: 1778224814412@@100.80.26.90@11000@ss_seu_df

SHOW tables;

SELECT COUNT(*) FROM dataset_multicity_20230917;

SELECT DISTINCT(uid) FROM dataset_multicity_20230917;

/opt/hive/bin/beeline \
  -u "jdbc:hive2://localhost:11000/default" \
  -n administrator \
  --silent=true \
  --showHeader=true \
  --outputformat=csv2 \
  -e "USE ss_seu_df; SELECT * FROM dataset_multicity_20230917_processed;" \
  > /mnt/d/MQ/DaaSBI/dataset_multicity_20230917_processed.csv \
  2> /tmp/hive_export_dataset_multicity_20230917_processed.log