import sqlite3, os

db = "mlflow.db"
old_base = "/Users/johbaum/code/autoresearcher-litter-detection"
new_base = "./mlruns"  # relative to where you run MLflow from

conn = sqlite3.connect(db)
cur = conn.cursor()

# Fix experiment artifact_location
cur.execute("SELECT experiment_id, artifact_location FROM experiments")
for exp_id, loc in cur.fetchall():
    if loc and old_base in loc:
        new_loc = loc.replace(old_base, "").lstrip("/")  # e.g. "mlruns/1"
        cur.execute("UPDATE experiments SET artifact_location=? WHERE experiment_id=?",
                    (new_loc, exp_id))

# Fix run artifact_uri
cur.execute("SELECT run_uuid, artifact_uri FROM runs")
for run_id, uri in cur.fetchall():
    if uri and old_base in uri:
        new_uri = uri.replace(old_base, "").lstrip("/")  # e.g. "mlruns/1/<id>/artifacts"
        cur.execute("UPDATE runs SET artifact_uri=? WHERE run_uuid=?",
                    (new_uri, run_id))

conn.commit()
conn.close()
print("Done")
