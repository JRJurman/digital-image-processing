import cv2

def flush():
   print('Press "c" to continue, ESC to exit')
   delay = 100
   while True:
      k = cv2.waitKey(delay)

      # ESC pressed
      if k == 27 or k == (65536 + 27):
         action = 'exit'
         print('Exiting ...')
         break

      # c or C pressed
      if k == 99 or k == (65536 + 99) or k == 67 or k == (65536 + 67):
         action = 'continue'
         print('Continuing ...')
         break

   return action
