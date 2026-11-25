from pathlib import Path
import secrets
import click
import sqlite3
from sqlalchemy import event
from sqlalchemy.engine import Engine
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_mail import Mail
from flask_wtf.csrf import CSRFProtect
from config import Config

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'routes.login'
login_manager.login_message = 'Увійдіть, щоб відкрити цю сторінку.'
mail = Mail()
csrf = CSRFProtect()

@event.listens_for(Engine, 'connect')
def enable_sqlite_constraints(connection, connection_record):
    if isinstance(connection, sqlite3.Connection):
        cursor = connection.cursor()
        cursor.execute('PRAGMA foreign_keys = ON')
        cursor.execute('PRAGMA busy_timeout = 5000')
        cursor.close()

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config)
    if test_config is not None:
        app.config.update(test_config)
    instance = Path(app.instance_path)
    instance.mkdir(parents=True, exist_ok=True)
    if not app.config.get('SECRET_KEY'):
        key_file = instance / '.secret_key'
        try:
            with key_file.open('x', encoding='ascii') as stream:
                stream.write(secrets.token_hex(32))
        except FileExistsError:
            pass
        app.config['SECRET_KEY'] = key_file.read_text(encoding='ascii').strip()
    if not app.config['MAIL_ENABLED']:
        app.config['MAIL_SUPPRESS_SEND'] = True
    db.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)
    csrf.init_app(app)
    from app.routes import routes
    app.register_blueprint(routes)
    from app.time_utils import local_format, form_datetime
    app.jinja_env.filters['local_dt'] = local_format
    app.jinja_env.filters['form_dt'] = form_datetime

    @app.cli.command('init-db')
    def init_db():
        from app.database import initialize_database
        backup = initialize_database()
        click.echo('Базу даних підготовлено: схема 2.3, версія 1.')
        if backup:
            click.echo(f'Резервна копія перед перенесенням: {backup}')

    @app.cli.command('reminders')
    def reminders():
        from app.utils import check_and_send_reminders
        click.echo(check_and_send_reminders(app))

    # Startup never launches a mail worker or updates the database schema.
    return app
