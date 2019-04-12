from abc import *

# 카메라가 여러 대인 경우에 observer 패턴 사용

class Observerable:
    def __init__(self):
        self.observers = []
    
    def addObserver(self, observer):
        self.observers.append(observer)
    
    def deleteObserver(self, observer):
        self.observers.remove(observer)

    def notifyAll(self, data):
        for observer in self.observers:
            observer.update(data)   # 추상클래스로 부터 data 전달 받음
        

# Observer 추상 클래스
class Observer(metaclass=ABCMeta):
    @abstractmethod
    def update(self, data):
        pass