from PyQt5.QtCore import QDateTime, Qt, QTimer

def add_seconds_to_date(time_string, seconds_to_add):
    # Chuyển đổi chuỗi thời gian thành QDateTime
    start_date = QDateTime.fromString(time_string, Qt.ISODate)

    # Thêm số giây vào start_date
    target_date = start_date.addSecs(seconds_to_add)

    # Trả về target_date dưới dạng chuỗi ISODate
    return target_date.toString(Qt.ISODate)

# Sử dụng hàm
start_time = "2023-12-23T12:00:00"  # Thay thế bằng giá trị thời gian thực tế
seconds_to_add = 3600  # Thay thế bằng số giây muốn thêm

result = add_seconds_to_date(start_time, seconds_to_add)
print("Start Date:", start_time)
print("Target Date:", result)