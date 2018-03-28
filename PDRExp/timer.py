import time
from threading import Timer,Thread,Event


class perpetualTimer():

   def __init__(self,t,hFunction):
      self.t=t
      self.hFunction = hFunction
      self.thread = Timer(self.t,self.handle_function)

   def handle_function(self):
      self.hFunction()
      self.thread = Timer(self.t,self.handle_function)
      self.thread.start()

   def start(self):
      self.thread.start()

   def cancel(self):
      self.thread.cancel()

def printer():
    print str(int(round(time.time() * 1000)))

t = perpetualTimer(0.1,printer)
while True:
    startTime =int(round(time.time() * 1000))
    if (startTime%1000) ==0:
	t.start()
	break

