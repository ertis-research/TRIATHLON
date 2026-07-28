import urllib.parse
import requests
import json
import gzip
import re

from datetime import datetime
from Crypto.Util.py3compat import *
from base64 import *
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

def pad(data_to_pad, block_size, style='pkcs7'):
    padding_len = block_size-len(data_to_pad)%block_size
    if style == 'pkcs7':
        padding = bchr(padding_len)*padding_len
    elif style == 'x923':
        padding = bchr(0)*(padding_len-1) + bchr(padding_len)
    elif style == 'iso7816':
        padding = bchr(128) + bchr(0)*(padding_len-1)
    else:
        raise ValueError("Unknown padding style")
    return data_to_pad + padding

def unpad(padded_data, block_size, style='pkcs7'):
    pdata_len = len(padded_data)
    if pdata_len % block_size:
        raise ValueError("Input data is not padded")
    if style in ('pkcs7', 'x923'):
        padding_len = bord(padded_data[-1])
        if padding_len<1 or padding_len>min(block_size, pdata_len):
            raise ValueError("Padding is incorrect.")
        if style == 'pkcs7':
            if padded_data[-padding_len:]!=bchr(padding_len)*padding_len:
                raise ValueError("PKCS#7 padding is incorrect.")
        else:
            if padded_data[-padding_len:-1]!=bchr(0)*(padding_len-1):
                raise ValueError("ANSI X.923 padding is incorrect.")
    elif style == 'iso7816':
        padding_len = pdata_len - padded_data.rfind(bchr(128))
        if padding_len<1 or padding_len>min(block_size, pdata_len):
            raise ValueError("Padding is incorrect.")
        if padding_len>1 and padded_data[1-padding_len:]!=bchr(0)*(padding_len-1):
            raise ValueError("ISO 7816-4 padding is incorrect.")
    else:
        raise ValueError("Unknown padding style")
    return padded_data[:-padding_len]

def IsNullOrWhiteSpace(str=None):
    return not str or str.isspace()

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime)):
            return obj.strftime('%d-%m-%Y;%H:%M:%S.%f')[:-3]

class CustomDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.try_datetime, *args, **kwargs)

    @staticmethod
    def try_datetime(d):
        ret = {}
        for key, value in d.items():
            try:
                ret[key] = datetime.strptime(value, '%d-%m-%Y;%H:%M:%S.%f')
            except (ValueError, TypeError):
                ret[key] = value
        return ret

class HTTPResponse:
    statusCode = ""
    reason = ""
    headers = {}
    payload = ""

class APIRequest:
    method = ""
    url = ""
    header = {}
    data = ""

class Historical:
    DateOfRecord: datetime
    Value: float
    Reason: str
    ChannelID: int

    def __init__(self, DateOfRecord: datetime, ChannelID: int, Reason: str, Value: float):
        self.DateOfRecord = DateOfRecord
        self.ChannelID = ChannelID
        self.Reason = Reason
        self.Value = Value

class Alarms:
    StationID: str
    DateOfRecord: datetime
    Reason: int
    ChannelID: int
    AlarmText: str
    RawValue: str

    def __init__(self, StationID: str, DateORecord: datetime, Reason: int, ChannelID: int, AlarmText: str, RawValue: str):
        self.StationID = StationID
        self.DateOfRecord = DateORecord
        self.Reason = Reason
        self.ChannelID = ChannelID
        self.AlarmText = AlarmText
        self.RawValue = RawValue

class LastKnownValues:
    lastConnectionTime: datetime
    RSSI: int
    PowerSupply: str
    values = {}

    def __init__(self, lastConnectionTime: datetime, RSSI: int, PowerSupply: str, values):
        self.lastConnectionTime = lastConnectionTime
        self.RSSI = RSSI
        self.PowerSupply = PowerSupply
        self.values = values

class ProgrammableRelayPeriod:
    WeekDays = []
    ActivationTime: str
    Duration: int

    def __init__(self, Weekdays, ActivationTime, Duration):
        self.WeekDays = Weekdays
        self.ActivationTime = ActivationTime
        self.Duration = Duration

class Client:

    IP = ""
    Port = None
    AuthAccount = ""
    Password = ""
    AuthPass = ""
    AuthToken = ""
    Created = False
    SSL = None
    ZeusAPILastError = ""
    APIVersion = "v1"

    def __init__(self, IP, Port, SSL, AuthAccount, Password, AuthPass):

        if IsNullOrWhiteSpace(IP) or IsNullOrWhiteSpace(Port) or IsNullOrWhiteSpace(AuthAccount) or IsNullOrWhiteSpace(AuthPass):
            self.ZeusAPILastError = "The input parameters are incorrect."
            return None

        self.IP = IP
        self.Port = Port
        self.AuthAccount = AuthAccount
        self.AuthPass = b64decode(AuthPass.encode('utf-8'))
        self.SSL = SSL

        if IsNullOrWhiteSpace(Password):
            self.Password = "fJFYD42CMLEEKsUe"
        else:
            self.Password = Password

        if (self.RequestAuthenticationToken() == False):
            self.IP = ""
            self.Port = None
            self.AuthAccount = ""
            self.Password = ""
            self.AuthPass = ""
            self.AuthToken = ""
            self.SSL = None
            self.Created = False
        else:
            self.Created = True

    def IsCreated(self):
        return self.Created

    def GetZeusApiLastError(self):
        return self.ZeusAPILastError

    def Stations(self):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return None

        result = None
        processResponse = self.CreateSendReceive("GET", "stations?status=all&Accept-Encoding=gzip", "", True, False, True)

        if processResponse == None:
            return None
        else:
            if processResponse.statusCode == "200":
                if IsNullOrWhiteSpace(processResponse.payload) == True:
                    self.ZeusAPILastError = "Incorrect data received."
                    return None
                else:
                    result = json.loads(processResponse.payload)
                    self.ZeusAPILastError = ""
        return result

    def OnlineStation(self):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return None

        result = None
        processResponse = self.CreateSendReceive("GET", "stations?status=online&Accept-Encoding=gzip", "", True, False, True)

        if processResponse == None:
            return None
        else:
            if processResponse.statusCode == "200":
                if IsNullOrWhiteSpace(processResponse.payload) == True:
                    self.ZeusAPILastError = "Incorrect data received."
                    return None
                else:
                    result = json.loads(processResponse.payload)
                    self.ZeusAPILastError = ""
        return result

    def PendingMessages(self):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return None

        result = None
        processResponse = self.CreateSendReceive("GET", "stations/pending-messages?Accept-Encoding=gzip", "", True, False, True)

        if processResponse == None:
            return None
        else:
            if processResponse.statusCode == "200":
                if IsNullOrWhiteSpace(processResponse.payload) == True:
                    self.ZeusAPILastError = "Incorrect data received."
                    return None
                else:
                    result = json.loads(processResponse.payload)
                    self.ZeusAPILastError = ""
        return result

    def GetStationProperties(self, stationID: str):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return None

        if IsNullOrWhiteSpace(stationID):
            self.ZeusAPILastError = "Incorrect input values: stationID."
            return None

        result = None
        processResponse = self.CreateSendReceive("GET", "stations/" + urllib.parse.quote_plus(stationID) + "/properties?Accept-Encoding=gzip", "", True, False, True)

        if processResponse == None:
            return None
        else:
            if processResponse.statusCode == "200":
                if IsNullOrWhiteSpace(processResponse.payload) == True:
                    self.ZeusAPILastError = "Incorrect data received."
                    return None
                else:
                    result = json.loads(processResponse.payload)
                    self.ZeusAPILastError = ""
        return result

    def GetAllStationProperties(self):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return None

        result = None
        processResponse = self.CreateSendReceive("GET", "stations/properties?Accept-Encoding=gzip", "", True, False, True)

        if processResponse == None:
            return None
        else:
            if processResponse.statusCode == "200":
                if IsNullOrWhiteSpace(processResponse.payload) == True:
                    self.ZeusAPILastError = "Incorrect data received."
                    return None
                else:
                    result = json.loads(processResponse.payload)
                    self.ZeusAPILastError = ""
        return result

    def GetStationExtendedProperties(self, stationID: str):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return None

        if IsNullOrWhiteSpace(stationID):
            self.ZeusAPILastError = "Incorrect input values: stationID."
            return None

        result = None
        processResponse = self.CreateSendReceive("GET", "stations/" + urllib.parse.quote_plus(stationID) + "/extendedproperties?Accept-Encoding=gzip", "", True, False, True)

        if processResponse == None:
            return None
        else:
            if processResponse.statusCode == "200":
                if IsNullOrWhiteSpace(processResponse.payload) == True:
                    self.ZeusAPILastError = "Incorrect data received."
                    return None
                else:
                    result = json.loads(processResponse.payload)
                    self.ZeusAPILastError = ""
        return result

    def GetAllStationExtendedProperties(self):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return None

        result = None
        processResponse = self.CreateSendReceive("GET", "stations/extendedproperties?Accept-Encoding=gzip", "", True, False, True)

        if processResponse == None:
            return None
        else:
            if processResponse.statusCode == "200":
                if IsNullOrWhiteSpace(processResponse.payload) == True:
                    self.ZeusAPILastError = "Incorrect data received."
                    return None
                else:
                    result = json.loads(processResponse.payload)
                    self.ZeusAPILastError = ""
        return result

    def GetHistorical(self, stationID: str, startDate: datetime, endDate: datetime, channels):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return None

        if IsNullOrWhiteSpace(stationID):
            self.ZeusAPILastError = "Incorrect input values: stationID."
            return None

        if channels == None or len(channels) == 0:
            self.ZeusAPILastError = "Incorrect input values: Unspecified channels."
            return None

        channelView = "CUSTOM"
        customChanelsQuery = ""

        for c in channels:
            customChanelsQuery = customChanelsQuery + str(c) + ","

        customChanelsQuery = customChanelsQuery.rstrip(",")
        correctURI = False
        uri = "historical/" + urllib.parse.quote_plus(stationID) + "?Accept-Encoding=gzip&startDate=" + startDate.strftime("%d-%m-%Y;%H:%M:%S") + "&endDate=" + endDate.strftime("%d-%m-%Y;%H:%M:%S") + "&channelView="
        result = None

        if IsNullOrWhiteSpace(channelView) == False:
            if channelView == "CUSTOM":
                if IsNullOrWhiteSpace(customChanelsQuery) == False:
                    uri = uri + "CUSTOM&channels=" + customChanelsQuery
                    correctURI = True
                else:
                    correctURI = False
            else:
                uri = uri + channelView
                correctURI = True
        else:
            correctURI = False

        if correctURI == True:
            processResponse = self.CreateSendReceive("GET", uri, "", True, False, True)

            if processResponse == None:
                return None
            else:
                if processResponse.statusCode == "200":
                    if IsNullOrWhiteSpace(processResponse.payload) == True:
                        self.ZeusAPILastError = "Incorrect data received."
                        return None
                    else:
                        result = json.loads(processResponse.payload, cls=CustomDecoder)
                        self.ZeusAPILastError = ""
        else:
            self.ZeusAPILastError = "Incorrect URI data."
            return None

        return result

    def GetAlarm(self, startDate: datetime, endDate: datetime, sinceThisRawValue: str = ""):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return None

        result = None
        correctURI = False
        uri = "alarms?Accept-Encoding=gzip&"

        if IsNullOrWhiteSpace(sinceThisRawValue) == True:
            uri = uri + "startDate=" + startDate.strftime("%d-%m-%Y;%H:%M:%S") + "&endDate=" + endDate.strftime("%d-%m-%Y;%H:%M:%S")
            correctURI = True
        else:
            uri = uri + "rawAlarm=" + sinceThisRawValue
            correctURI = True


        if correctURI == True:
            processResponse = self.CreateSendReceive("GET", uri, "", True, False, True)

            if processResponse == None:
                return None
            else:
                if processResponse.statusCode == "200":
                    if IsNullOrWhiteSpace(processResponse.payload) == True:
                        self.ZeusAPILastError = "Incorrect data received."
                        return None
                    else:
                        result = json.loads(processResponse.payload, cls=CustomDecoder)
                        self.ZeusAPILastError = ""
        else:
            self.ZeusAPILastError = "Incorrect URI data."
            return None

        return result

    def GetLastKnownValues(self, stationID: str):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return None

        if IsNullOrWhiteSpace(stationID) == True:
            self.ZeusAPILastError = "Incorrect input values: stationID."
            return None

        result = None
        processResponse = self.CreateSendReceive("GET", "stations/" + urllib.parse.quote_plus(stationID) + "/lastknownvalues?Accept-Encoding=gzip", "", True, False, True)

        if processResponse == None:
            return None
        else:
            if processResponse.statusCode == "200":
                if IsNullOrWhiteSpace(processResponse.payload) == True:
                    self.ZeusAPILastError = "Incorrect data received."
                    return None
                else:
                    result = json.loads(processResponse.payload, cls=CustomDecoder)
                    self.ZeusAPILastError = ""

        return result

    def GetLastKnownValuesExtended(self, stationID: str):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return None

        if IsNullOrWhiteSpace(stationID):
            self.ZeusAPILastError = "Incorrect input values: stationID."
            return None

        result = None
        processResponse = self.CreateSendReceive("GET", "stations/" + urllib.parse.quote_plus(stationID) + "/lastknownvaluesExtended?Accept-Encoding=gzip", "", True, False, True)

        if processResponse == None:
            return None
        else:
            if processResponse.statusCode == "200":
                if IsNullOrWhiteSpace(processResponse.payload) == True:
                    self.ZeusAPILastError = "Incorrect data received."
                    return None
                else:
                    result = json.loads(processResponse.payload, cls=CustomDecoder)
                    self.ZeusAPILastError = ""
        return result

    def GetAllStationLastKnownValues(self):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return None

        result = None
        processResponse = self.CreateSendReceive("GET", "stations/lastknownvalues?Accept-Encoding=gzip", "", True, False, True)

        if processResponse == None:
            return None
        else:
            if processResponse.statusCode == "200":
                if IsNullOrWhiteSpace(processResponse.payload) == True:
                    self.ZeusAPILastError = "Incorrect data received."
                    return None
                else:
                    result = json.loads(processResponse.payload, cls=CustomDecoder)
                    self.ZeusAPILastError = ""

        return result

    def GetAllStationLastKnownValuesExtended(self):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return None

        result = None
        processResponse = self.CreateSendReceive("GET", "stations/lastknownvaluesExtended?Accept-Encoding=gzip", "", True, False, True)

        if processResponse == None:
            return None
        else:
            if processResponse.statusCode == "200":
                if IsNullOrWhiteSpace(processResponse.payload) == True:
                    self.ZeusAPILastError = "Incorrect data received."
                    return None
                else:
                    result = json.loads(processResponse.payload, cls=CustomDecoder)
                    self.ZeusAPILastError = ""

        return result

    def GetConfigurationRelay(self, stationID: str, output: int):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return None

        if IsNullOrWhiteSpace(stationID) == True or output < 0:
            self.ZeusAPILastError = "Incorrect input values: stationID | output."
            return None

        result = None
        processResponse = self.CreateSendReceive("GET", "stations/" + urllib.parse.quote_plus(stationID) + "/configurationRelay?Accept-Encoding=gzip&output=" + str(output), "", True, False, True)

        if processResponse == None:
            return None
        else:
            if processResponse.statusCode == "200":
                if IsNullOrWhiteSpace(processResponse.payload) == True:
                    self.ZeusAPILastError = "Incorrect data received."
                    return None
                else:
                    result = json.loads(processResponse.payload)
                    self.ZeusAPILastError = ""

        return result

    def SendMessage(self, destination: str, messageText: str, sendBySMS: bool = False):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return False

        if IsNullOrWhiteSpace(destination) == True or IsNullOrWhiteSpace(messageText) == True:
            self.ZeusAPILastError = "Incorrect input values: String."
            return False

        jsonDataString = json.dumps({ "Key" : destination, "Value" : messageText})

        if IsNullOrWhiteSpace(jsonDataString) == True:
            self.ZeusAPILastError = "Incorrect input values: destination | messageText."
            return False

        result = False
        uri = "messages/message?Content-Encoding=gzip&bySMS="
        if sendBySMS == True:
            uri = uri + "TRUE"
        else:
            uri = uri + "FALSE"
        uri = uri + "&dontWaitForResponse=FALSE"

        processResponse = self.CreateSendReceive("POST", uri, jsonDataString, True, True, True)

        if processResponse == None:
            result = False
        else:
            if processResponse.statusCode == "200":
                self.ZeusAPILastError = ""
                result = True
            else:
                result = False

        return result

    def SetHistorical(self, stationID: str, historicalValues):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return False

        if IsNullOrWhiteSpace(stationID) == True or historicalValues == None or len(historicalValues) == 0:
            self.ZeusAPILastError = "Incorrect input values."
            return False

        jsonDataString = json.dumps([ob.__dict__ for ob in historicalValues],  cls=CustomEncoder)

        if IsNullOrWhiteSpace(jsonDataString) == True:
            self.ZeusAPILastError = "Incorrect input values: historicalValues."
            return False

        result = False
        processResponse = self.CreateSendReceive("POST", "historical/" + urllib.parse.quote_plus(stationID) + "?Content-Encoding=gzip", jsonDataString, True, True, True)

        if processResponse == None:
            result = False
        else:
            if processResponse.statusCode == "200":
                self.ZeusAPILastError = ""
                result = True
            else:
                result = False

        return result

    def SetAlarms(self, alarmValues):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return False

        if alarmValues == None or len(alarmValues) == 0:
            self.ZeusAPILastError = "Incorrect input values."
            return False

        jsonDataString = json.dumps([ob.__dict__ for ob in alarmValues],  cls=CustomEncoder)

        if IsNullOrWhiteSpace(jsonDataString) == True:
            self.ZeusAPILastError = "Incorrect input values: alarmValues."
            return False

        result = False
        processResponse = self.CreateSendReceive("POST", "alarms?Content-Encoding=gzip&rawAlarm=FALSE", jsonDataString, True, True, True)

        if processResponse == None:
            result = False
        else:
            if processResponse.statusCode == "200":
                self.ZeusAPILastError = ""
                result = True
            else:
                result = False

        return result

    def SetLastKnownValues(self, stationID: str, values):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return False

        if IsNullOrWhiteSpace(stationID) or values == None:
            self.ZeusAPILastError = "Incorrect input values."
            return False

        jsonDataString = json.dumps(values.__dict__,  cls=CustomEncoder)

        if IsNullOrWhiteSpace(jsonDataString) == True:
            self.ZeusAPILastError = "Incorrect input values: alarmValues."
            return False

        result = False
        processResponse = self.CreateSendReceive("POST", "stations/" + urllib.parse.quote_plus(stationID)  + "/lastknownvalues?Content-Encoding=gzip", jsonDataString, True, True, True)

        if processResponse == None:
            result = False
        else:
            if processResponse.statusCode == "200":
                self.ZeusAPILastError = ""
                result = True
            else:
                result = False

        return result

    def SetLastKnownValuesExtended(self, stationID: str, values):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return False

        if IsNullOrWhiteSpace(stationID) or values == None:
            self.ZeusAPILastError = "Incorrect input values."
            return False

        jsonDataString = json.dumps(values,  cls=CustomEncoder)

        if IsNullOrWhiteSpace(jsonDataString) == True:
            self.ZeusAPILastError = "Incorrect input values: values."
            return False

        result = False
        processResponse = self.CreateSendReceive("POST", "stations/" + urllib.parse.quote_plus(stationID)  + "/lastknownvaluesextended?Content-Encoding=gzip", jsonDataString, True, True, True)

        if processResponse == None:
            result = False
        else:
            if processResponse.statusCode == "200":
                self.ZeusAPILastError = ""
                result = True
            else:
                result = False

        return result

    def SetConfigurationRelay(self, destination: str, output: int, myProgrammableRelayPeriod):
        if self.Created == False:
            self.ZeusAPILastError = "Object Client not created."
            return False

        if IsNullOrWhiteSpace(destination) == True or output < 0 or myProgrammableRelayPeriod == None:
            self.ZeusAPILastError = "Incorrect input: destination | output | myProgrammableRelayPeriod empty."
            return False

        if len(myProgrammableRelayPeriod) > 4:
            self.ZeusAPILastError = "Incorrect input values: myProgrammableRelayPeriod elements."
            return False

        # Create the command
        messageText = "PR" + str(output) + "="
        for element in myProgrammableRelayPeriod:
            messageText = messageText + element.ActivationTime + "-" + str(element.Duration) + "@"
            for myDay in element.WeekDays:
                if re.search("^[LMXJVSD]+$", myDay) == None:
                    self.ZeusAPILastError = "Incorrect input values: myProgrammableRelayPeriod.Weekdays format."
                    return False
                else:
                    messageText = messageText + myDay
            messageText = messageText + " "

        for x in range(len(myProgrammableRelayPeriod), 4):
            messageText = messageText + "00:00-0@ "

        messageText = messageText.rstrip()

        jsonDataString = json.dumps({"Key" : destination, "Value" : messageText})

        if IsNullOrWhiteSpace(jsonDataString) == True:
            self.ZeusAPILastError = "Incorrect input values: destination | messageText."
            return False

        result = False
        processResponse = self.CreateSendReceive("POST", "messages/message?Content-Encoding=gzip&bySMS=FALSE&dontWaitForResponse=TRUE", jsonDataString, True, True, True)

        if processResponse == None:
            result = False
        else:
            if processResponse.statusCode == "200":
                self.ZeusAPILastError = ""
                result = True
            else:
                result = False

        return result

    def RequestAuthenticationToken(self):

        result = False

        encryptedPass = self.EncryptString_Aes(self.Password)
        if IsNullOrWhiteSpace(encryptedPass):
            self.ZeusAPILastError = "The data could not be encrypted."
            self.AuthToken = ""
            return False

        processResponse = self.CreateSendReceive("GET", "accounts/" + urllib.parse.quote_plus(self.AuthAccount) + "?authentication=" + urllib.parse.quote_plus(encryptedPass), "", False, False, False, 1, True)
        if processResponse == None:
            self.AuthToken = ""
            result = False
        else:
            if processResponse.statusCode == "200":
                if IsNullOrWhiteSpace(processResponse.payload) == False:
                    self.AuthToken = processResponse.payload
                    result = True
                else:
                    self.AuthToken = ""
                    result = False
            else:
                self.AuthToken = ""
                result = False

        return result

    def CreateSendReceive(self, method, resource, payload, AcceptEncoding, ContentEncoding, OptionDecrypt, Attempts = 3, AuthRequest = False):
        myResponse = HTTPResponse()
        intentos = 0

        while True:
            # Creamos la peticion
            myRequest = self.CreateRequest(method, resource, payload, AcceptEncoding, ContentEncoding)
            if myRequest == None:
                self.ZeusAPILastError = "Creating request failure."
                return None

            # Enviamos la peticion
            respondRequest = requests.request(myRequest.method, myRequest.url, headers=myRequest.header, data = myRequest.data)

            # Procesamos la peticion
            myResponse = self.ProccessRespond(respondRequest, OptionDecrypt, AcceptEncoding)
            if myResponse == None:
                intentos = Attempts
                self.ZeusAPILastError = "Proccess ZeusServer Respond failure."
                return None
            else:
                if myResponse.statusCode == "200":
                    intentos = Attempts
                    self.ZeusAPILastError = ""
                elif myResponse.statusCode == "401":
                    self.ZeusAPILastError = "Request Authentication failure."
                    if AuthRequest == False:
                        intentos = intentos + 1
                        myResponse = None
                        self.RequestAuthenticationToken()
                    else:
                        myResponse = None
                        intentos = Attempts
                elif myResponse.statusCode == "400":
                    self.ZeusAPILastError = self.ZeusAPILastError + "Reason: Bad Request."
                    if IsNullOrWhiteSpace(myResponse.payload) == False:
                        self.ZeusAPILastError = self.ZeusAPILastError + " -- Server message: " + myResponse.payload
                    intentos = Attempts
                    myResponse = None
                elif myResponse.statusCode == "403":
                    self.ZeusAPILastError = self.ZeusAPILastError + "Reason: Bad Request."
                    if IsNullOrWhiteSpace(myResponse.payload) == False:
                        self.ZeusAPILastError = self.ZeusAPILastError + " -- Server message: " + myResponse.payload
                    intentos = Attempts
                    myResponse = None
                elif myResponse.statusCode == "404":
                    self.ZeusAPILastError = self.ZeusAPILastError + "Reason: The requested resource was not found."
                    if IsNullOrWhiteSpace(myResponse.payload) == False:
                        self.ZeusAPILastError = self.ZeusAPILastError + " -- Server message: " + myResponse.payload
                    intentos = Attempts
                    myResponse = None
                elif myResponse.statusCode == "405":
                    self.ZeusAPILastError = self.ZeusAPILastError + "Reason: The method is not allowed."
                    if IsNullOrWhiteSpace(myResponse.payload) == False:
                        self.ZeusAPILastError = self.ZeusAPILastError + " -- Server message: " + myResponse.payload
                    intentos = Attempts
                    myResponse = None
                elif myResponse.statusCode == "500":
                    self.ZeusAPILastError = self.ZeusAPILastError + "Reason: An internal error occurred on the server."
                    if IsNullOrWhiteSpace(myResponse.payload) == False:
                        self.ZeusAPILastError = self.ZeusAPILastError + " -- Server message: " + myResponse.payload
                    intentos = Attempts
                    myResponse = None
                else:
                    self.ZeusAPILastError = self.ZeusAPILastError + "Reason: Unknown error."
                    if IsNullOrWhiteSpace(myResponse.payload) == False:
                        self.ZeusAPILastError = self.ZeusAPILastError + " -- Server message: " + myResponse.payload
                    intentos = Attempts
                    myResponse = None
            if intentos >= Attempts:
                break
        return myResponse

    def CreateRequest(self, method, resource, payload, AcceptEncoding, ContentEncoding):
        request = APIRequest()
        if method == "GET" or method == "POST" or method == "DELETE" or method == "PUT":
            request.method = method
            myHttp = "http"
            if self.SSL == True:
                myHttp = myHttp + "s"
            request.url = myHttp + "://" + self.IP + ":" + self.Port + "/api/" + self.APIVersion + "/" + resource
            request.headers = {
                "User-Agent" : "ZeusClient ApiREST",
                "Date" : self.httpdate(datetime.utcnow())
            }
            # Si tenemos AuthToken lo añadimos como cabecera.
            if IsNullOrWhiteSpace(self.AuthToken) == False:
                request.header["Authorization"] = "Bearer " + self.AuthToken
            # Si tenemos payload que enviar
            if IsNullOrWhiteSpace(payload) == False:
                newPayload = ""
                # Si hay que comprimir el payload
                if ContentEncoding == True:
                    newPayload = self.CompressString(payload)
                else:
                    newPayload = payload
                # Encriptamos el payload
                encryptedPayload = self.EncryptString_Aes(newPayload)
                if IsNullOrWhiteSpace(encryptedPayload):
                    return None
                else:
                    request.data = encryptedPayload
                    return request
            else:
                return request
        else:
            return None

    def ProccessRespond(self, httpRespond, decrypt, decompress = False):
        myResponse = HTTPResponse()
        if httpRespond == None:
            return None
        else:
            myResponse.statusCode = str(httpRespond.status_code)
            myResponse.reason = str(httpRespond.reason)
            myResponse.headers = httpRespond.headers
            payload = httpRespond.text

            if myResponse.statusCode == "200":
                if IsNullOrWhiteSpace(payload) == False:
                    # Decrypt
                    decryptPayload = ""
                    if decrypt == True:
                        decryptPayload = self.DecryptString_Aes(payload)
                        if IsNullOrWhiteSpace(decryptPayload) == True:
                            return None
                    else:
                        decryptPayload = payload
                    #GZIP
                    if decompress == True:
                        myResponse.payload = self.DecompressString(decryptPayload)
                    else:
                        myResponse.payload = decryptPayload
            else:
                myResponse.payload = payload
        return myResponse

    def httpdate(self, dt):
            """Return a string representation of a date according to RFC 1123
            (HTTP/1.1).

            The supplied date must be in UTC.

            """
            weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dt.weekday()]
            month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][dt.month - 1]
            return "%s, %02d %s %04d %02d:%02d:%02d GMT" % (weekday, dt.day, month, dt.year, dt.hour, dt.minute, dt.second)

    def EncryptString_Aes(self, payloadToEncrypt):

        if IsNullOrWhiteSpace(payloadToEncrypt):
            return ""

        # print(type(payloadToEncrypt))

        # Get bytes
        data_bytes = payloadToEncrypt.encode('utf-8')

        # Generate vector IV
        IV_bytes = get_random_bytes(16)

        try:
            # Encrypt
            cipher = AES.new(self.AuthPass, AES.MODE_CBC, IV_bytes)
            ct_bytes = cipher.encrypt(pad(data_bytes, AES.block_size))
            return b64encode(IV_bytes).decode('utf-8') + b64encode(ct_bytes).decode('utf-8')
        except ValueError:
            return ""

    def DecryptString_Aes(self, encryptedString):

        if IsNullOrWhiteSpace(encryptedString):
            return ""

        # Get vector iv and data
        IV_string = encryptedString[0:24]
        data_string = encryptedString[24:]

        # Pass to bytes.
        IV_bytes = b64decode(IV_string.encode('utf-8'))
        data_bytes = b64decode(data_string.encode('utf-8'))

        # Decrypt
        try:
            cipher = AES.new(self.AuthPass, AES.MODE_CBC, IV_bytes)
            pt = unpad(cipher.decrypt(data_bytes), AES.block_size)
            return pt.decode('utf-8')
        except ValueError:
            return ""

    def CompressString(self, text: str):
        return (b64encode(gzip.compress(text.encode('utf-8')))).decode('utf-8')

    def DecompressString(self, compressedText: str):
        return (gzip.decompress(b64decode(compressedText))).decode('utf-8')
