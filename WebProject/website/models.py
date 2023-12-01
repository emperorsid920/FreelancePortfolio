from website import db                  #importing db from website package
from flask_login import UserMixin       #helps Users login
from sqlalchemy.sql import func

class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)                         #unique Indetifier
    data = db.Column(db.String(10000))                                   #Data associated with notes
    date = db.Column(db.DateTime(timezone=True), default=func.now())     #Store relevant date and time info
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))            #Reference to the User Model


#Database Model defination
#All the user info will be stored here
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)     #unique Indetifier
    email = db.Column(db.String(150), unique=True)   #unique email address
    password = db.Column(db.String(150))             #stores password
    first_name = db.Column(db.String(150))           #stores first name
    notes = db.relationship('Note')                  #List that will store all related notes
