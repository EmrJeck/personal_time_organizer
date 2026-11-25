from app import create_app
from app.utils import check_and_send_reminders

if __name__ == '__main__':
    print(check_and_send_reminders(create_app()))
