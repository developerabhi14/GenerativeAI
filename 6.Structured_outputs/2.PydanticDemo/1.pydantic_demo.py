from pydantic import BaseModel, EmailStr, Field
from typing import Optional
class Student(BaseModel):
    name: str
    age: int = 54
    profession:Optional[str]= None
    email: Optional[EmailStr]= None
    cgpa:Optional[float]= Field(default=None, gt=0.0, lt=10.0, description="A d") #ge--> greater than equal to, le--> less than equal to



# type cohercion --> Pydantic will try to convert the types if they don't match
new_student ={'name':"John Doe", 'age':'25'}

# default value

new_student1 ={'name':"John Doe"}

new_student2 ={'name':"John Doe", 'profession':"Engineer"}

new_student3 ={'name':"John Doe", 'profession':"Engineer", 'email':"abc@a.a"}

new_student4 ={'name':"John Doe", 'profession':"Engineer", 'email':"abc@gmail.com"}

new_student5 ={'name':"John Doe", 'profession':"Engineer", 'email':"abc@gmail.com",'cgpa':9}
student=Student(**new_student)
print(student)
print(type(student))

student1=Student(**new_student1)
print(student1)
print(type(student1))

print(student1.age)

student2=Student(**new_student2)
print(student2)

student3=Student(**new_student3)
print(student3)

student4=Student(**new_student4)
print(student4)

student5=Student(**new_student5)
print(student5)

student_dict=dict(student5)
print(student_dict["age"])

student_json=student5.model_dump_json()
print(student_json)