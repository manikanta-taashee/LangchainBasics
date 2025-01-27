import sqlite3
import os
from langchain.tools import Tool

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
dbpath = os.path.join(current_dir, "db.sqlite")
print(dbpath)

conn = sqlite3.connect(dbpath)

def list_tables():
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = cursor.fetchall()
    return "\n".join(row[0] for row in rows if row[0] is not None)

def run_sqlite_query(query):
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()


run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Useful when you need to run a SQL query",
    func=run_sqlite_query,
)

def describe_tables(table_names):
    c = conn.cursor()
    tables = ', '.join("'" + table + "'" for table in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables});")
    return '\n'.join(row[0] for row in rows if row[0] is not None)

describe_tables_tool = Tool.from_function(
    name="describe_tables",
    description="Given a list of table names, returns the schema of those tables",
    func=describe_tables,
)

