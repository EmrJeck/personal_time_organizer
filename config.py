"""Local master's project settings; no old SMTP credentials."""
import os

class Config:
    SECRET_KEY = os.environ.get('ORGANIZER_SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///site.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SESSION_COOKIE_NAME = 'masters_organizer_session'
    APP_TIMEZONE = 'Europe/Kyiv'
    MAIL_ENABLED = False
    MAIL_SUPPRESS_SEND = True
    MAIL_SERVER = 'localhost'
    MAIL_PORT = 25
    MAIL_USE_TLS = False
    MAIL_USE_SSL = False
    MAIL_USERNAME = None
    MAIL_PASSWORD = None
    MAIL_DEFAULT_SENDER = None
    MAIL_DEBUG = False
