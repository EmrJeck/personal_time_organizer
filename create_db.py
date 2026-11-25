from app import create_app
from app.database import initialize_database

if __name__ == '__main__':
    with create_app().app_context():
        initialize_database()
    print('Базу даних підготовлено.')
