import sqlite3
import os
from langchain.tools import Tool

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
dbpath = os.path.join(current_dir, "db.sqlite")
print(dbpath)

conn = sqlite3.connect(dbpath)

def run_sqlite_query(query):
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()


run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Useful when you need to run a SQL query",
    func=run_sqlite_query,
)
