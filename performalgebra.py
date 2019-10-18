import json
import requests
def sumofnumber(ar):
    return sum(ar)
def averageofnumber(ar):
    return (sum(ar)/len(ar))
if __name__ == "__main__":
    request= requests.get("http://---")
    request_txt=request.text
    data=json.loads(request_txt)
    ar=data["numbers"]
    perform=data["perform"]
    if perform=="sum":
       result=sumofnumber(ar)
    else:
       result=averageofnumber(ar)
    data_serialized=json.dump(result,open("result.json", "w"))
    
    
