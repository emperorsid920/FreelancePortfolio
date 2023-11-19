from . import db                        #importing db from website package
from flask_login import UserMixin       #helps Users login

 class User(db.Model, UserMixin):
