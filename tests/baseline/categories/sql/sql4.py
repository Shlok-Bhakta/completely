##(1, "John", 30)
import sqlite3

conn = sqlite3.connect(':memory:')
c = conn.cursor()

c.execute('''CREATE TABLE users
             (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)''')

c.execute("INSERT INTO users VALUES (?, ?, ?)",
          <cursor1>)