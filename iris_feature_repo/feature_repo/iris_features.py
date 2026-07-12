from datetime import timedelta
from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32

# 1. Entity: uniquely identifies each iris sample
iris_entity = Entity(
    name="iris_id",
    join_keys=["iris_id"],
    description="Unique identifier for each iris flower sample",
)

# 2. Data source: points to our parquet file
iris_source = FileSource(
    path="data/iris.parquet",
    timestamp_field="event_timestamp",
)

# 3. Feature view: maps the feature columns to the entity and data source
iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris_entity],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    source=iris_source,
)
