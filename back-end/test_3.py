import time
from datetime import datetime

# Lấy giá trị thời gian hiện tại
timestamp = time.time()

# Chuyển đổi thành đối tượng datetime
datetime_obj = datetime.fromtimestamp(timestamp)

print("Giá trị thời gian:", timestamp)
print("Đối tượng datetime:", datetime_obj)
