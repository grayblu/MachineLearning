from websocket import WebSocketApp
from threading import Thread
import time
import json

def on_message(ws, message):
    print(message)

def on_error(ws, error):
    print(error)

def on_close(ws):
    print('### closed ###')

def on_open(ws): 
    carMsg = {
        'msgType' : 'POSITION',
        'target' : 2,
        'lat' : 37.555882,
        'lng' : 126.969732
    }

    car_str = json.dumps(carMsg)
    def run(*args):
        ws.send(car_str)
        ws.close()
        print('thread terminating...')
    Thread(target = run).start()


if __name__=='__main__':
    ws = WebSocketApp('ws://localhost:8080/start/car',
                            on_open=on_open,
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close)
    
    ws.run_forever()    # 이벤트 처리 루프 -close되면 리턴
