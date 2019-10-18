import json
import requests
def sumofnumber(ar):
    return sum(ar)
def averageofnumber(ar):
    return (sum(ar)/len(ar))
if __name__ == "__main__":
    with open("sample.json","r") as read_it:
        data=json.load(read_it)
    ar=data["numbers"]
    perform=data["perform"]
    if perform=="sum":
        print(sumofnumber(ar))
    if perform=="average":
        print(averageofnumber(ar))
    