from flask import Flask

from .cli_db import db_cli
from .cli_statistics import statistics_cli
from .models import db

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
db.init_app(app)
app.cli.add_command(db_cli)
app.cli.add_command(statistics_cli)
with app.app_context():
    db.create_all()

from app import routes  # noqa: F401, E402
