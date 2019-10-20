# -*- coding: utf-8 -*-

"""
AI-PAAS ,Ryerson Univesity

Created on Sat Sep 21 13:31:23 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Student
    
"""
import sqlalchemy as sql_al
from   sqlalchemy.ext.declarative import declarative_base
import sqlalchemy.orm as sql_orm

Base = declarative_base()

class Aircraft_Engine(Base):
    
    __tablename__ = "aircraft_engines"
    
    id            = sql_al.Column(sql_al.Integer, primary_key=True)
    air_engine_no = sql_al.Column(sql_al.Integer)
    dataset_type  = sql_al.Column(sql_al.String)
    actual_RUL    = sql_al.Column(sql_al.Integer)      #Null for the train dataset
    
    sensor_measurements = sql_orm.relationship("Sensor_Measurement", back_populates = "aircraft_engine")
    opcond_measurements = sql_orm.relationship("OpCond_Measurement", back_populates = "aircraft_engine")
    
    
class Sensor(Base):
    
    __tablename__ = "sensors"
    
    id   = sql_al.Column(sql_al.Integer, primary_key=True)
    name = sql_al.Column(sql_al.String)
    
    sensor_measurements = sql_orm.relationship("Sensor_Measurement", back_populates = "sensor")
    
class Op_Cond(Base): #Operating Conditions
    
    __tablename__ = "op_conditions"
    
    id   = sql_al.Column(sql_al.Integer, primary_key=True)
    name = sql_al.Column(sql_al.String)
    
    opcond_measurements = sql_orm.relationship("OpCond_Measurement", back_populates = "op_cond")
    
class Sensor_Measurement(Base):
    
    __tablename__ = "sensor_meaurement"
    
    id            = sql_al.Column(sql_al.Integer, primary_key=True)
    value         = sql_al.Column(sql_al.Integer)   
    unit          = sql_al.Column(sql_al.String)
    air_engine_id = sql_al.Column(sql_al.Integer, sql_al.ForeignKey('aircraft_engines.id'))
    sensors_id    = sql_al.Column(sql_al.Integer, sql_al.ForeignKey('sensors.id'))
    
    sensor              = sql_orm.relationship("Sensor", back_populates = "sensor_measurements")
    aircraft_engine     = sql_orm.relationship("Aircraft_Engine", back_populates = "sensor_measurements")
    
class OpCond_Measurement(Base):
    
    __tablename__ = "opcond_measurements"
    
    id            = sql_al.Column(sql_al.Integer, primary_key=True)
    value         = sql_al.Column(sql_al.Integer)   
    unit          = sql_al.Column(sql_al.String)
    air_engine_id = sql_al.Column(sql_al.Integer, sql_al.ForeignKey('aircraft_engines.id'))
    opcond_id     = sql_al.Column(sql_al.Integer, sql_al.ForeignKey('op_conditions.id'))
    
    op_cond            = sql_orm.relationship("Op_Cond", back_populates = "opcond_measurements")
    aircraft_engine    = sql_orm.relationship("Aircraft_Engine", back_populates = "opcond_measurements")
    
    
def create_session(engine_address):
    engine = create_session(sql_al.create_engine(engine_address)),
    Base.metadata.create_all(engine)    
    Session = sql_orm.sessionmaker(bind=engine)
    session = Session()
    return session

if __name__ == '__main__':

    session = create_session(sql_al.create_engine('sqlite:///:memory:', echo=True))
    
    airengine_row = Aircraft_Engine(air_engine_no = 1, dataset_type = 'test', actual_RUL = 33) 
        
    session.add(airengine_row)    
    
    session.commit()
        
    for instance in session.query(Aircraft_Engine).order_by(Aircraft_Engine.id):
        print(instance.air_engine_no, instance.dataset_type, instance.actual_RUL)

    a = session.query(Aircraft_Engine).order_by(Aircraft_Engine.id).all()








