import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import telepot
import cv2
import logging

bot = telepot.Bot("6909646274:AAHI0EvBx-9XyTAvNXsorNsgyWI_V1Gsqp0") 

def get_file_extension(filename):
  return f"{filename}".split(".")[::-1][0]

def telegram_send_text_message(text_message, telegrams_id):
  try:
      
      formatted_message = text_message
      bot.sendMessage(telegrams_id, formatted_message)  

  except Exception as ex:
    raise Exception('Lỗi ', ex)
  
def telegram_send_image(img_path, telegrams_id, text_send):
  try:
    bot.sendPhoto(telegrams_id, photo=open(img_path, 'rb'), caption=text_send, parse_mode= 'Markdown')
  except Exception as ex:
    raise Exception('Lỗi ', ex)
  
if __name__ == '__main__':
#   for i in range(100):
#     telegram_send_text_message("Bạn xin lỗi đi!!!", 6592392299)
  # telegram_send_text_message("Bot chào buổi sáng", 5444005902)
    telegram_send_image(r"C:\Users\Admin\Desktop\photo_2023-12-18_08-37-44.jpg", 5444005902, "Bot nhận được phản hồi. Vui lòng bạn giải trình!")

