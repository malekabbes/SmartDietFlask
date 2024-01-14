 #calculating BMI
import numpy as np

def bmiCalculation(data):    
    age=int(data['age'])
    veg=float(data['vegnveg'])
    weight=float(data['weight'])
    height=float(data['height'])
    bmivalue = weight/((height/100)**2) 
    agecl = None  # Initialize agecl to None or some default value

    agewiseinp=0
        
    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        for i in test_list: 
            if(i == age):
                tr=round(lp/20)  
                agecl=round(lp/20)    

        
    #conditions
    print("Your body mass index is: ", bmivalue)
    if ( bmivalue < 16):
        print("Acoording to your BMI, you are Severely Underweight")
        clbmi=4
        val=0
        output= "Acoording to your BMI, you are Severely Underweight"
    elif ( bmivalue >= 16 and bmivalue < 18.5):
        print("Acoording to your BMI, you are Underweight")
        clbmi=3
        val=0
        output= "Acoording to your BMI, you are Underweight"
    elif ( bmivalue >= 18.5 and bmivalue < 25):
        print("Acoording to your BMI, you are Healthy")
        clbmi=2
        val=1
        output= "Acoording to your BMI, you are Healthy"
    elif ( bmivalue >= 25 and bmivalue < 30):
        print("Acoording to your BMI, you are Overweight")
        clbmi=1
        val=2
        output= "Acoording to your BMI, you are Overweight"
    elif ( bmivalue >=30): 
        print("Acoording to your BMI, you are Severely Overweight")
        clbmi=0
        val=3
        output= "Acoording to your BMI, you are Severely Overweight"
    return val,clbmi,agecl,veg,output,bmivalue
