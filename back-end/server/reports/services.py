from server.models import Video, Report, ReportImage
from server.extension import db_session
from flask import jsonify, request
from server.ma_schemas import VideosSchema, ReportsSchema, ImagesSchema
from sqlalchemy import desc, and_, or_
from datetime import datetime

videos_schema = VideosSchema(many=True)
reports_schema = ReportsSchema(many=True)
images_schema = ImagesSchema(many=True)

def add_video_service(path, start_time):
  session = db_session()
  try:
    t = int(datetime.now().timestamp())
    video = Video(path, t, start_time)
    session.add(video)
  
  except Exception as e:
    session.rollback()

    raise Exception({ 
      "message": "Thêm video vào db không thành công !", 
      "error": f"Error {e}"
    })
  
  finally:
    session.commit()
    session.close()

def get_videos_service():
  session = db_session()

  try:
    # params
    page_num = request.args.get("pageNum", type=int, default=1)
    page_size = request.args.get("pageSize", type=int, default=20)
    start_time = request.args.get("startTime", type=int, default=None)
    end_time = request.args.get("endTime", type=int, default=None)
    
    if start_time is None or end_time is None:
      return jsonify({"message": "Bad request !"}), 400

    else:
      videos = session.query(Video).filter(and_(
        Video.time >= start_time,
        Video.time <= end_time
      ))
      
      totalVideos = videos.count()

      # pagination
      offset = (page_num - 1) * page_size
      videos = videos.order_by(desc(Video.id))\
        .offset(offset).limit(page_size).all()

      return jsonify({
        "totalVideos": totalVideos,
        "videos": videos_schema.dump(videos)
      })

  except Exception as e:
    return jsonify({"message": f"Error {e}"}), 400
  
  finally:
    session.close()

def get_videos_path_db(page_num, page_size, start_time, end_time):
  session = db_session()

  try:
    if start_time is None or end_time is None:
      return jsonify({"message": "Bad request !"}), 400

    else:
      videos = session.query(Video).filter(and_(
        Video.time >= start_time,
        Video.time <= end_time
      ))
      
      totalVideos = videos.count()

      # pagination
      # offset = (page_num - 1) * page_size
      # videos = videos.order_by(desc(Video.id))\
      #   .offset(offset).limit(page_size).all()
      
      videos = videos.order_by(desc(Video.id)).all()
      

      return videos

  except Exception as e:
    print(f"[get_videos_path_db]: {e}")
    return []
  
  finally:
    session.close()

def add_report_service(path_video, person_name, age, gender, mask, code_color, time, images_path, is_front): 
  session = db_session()

  try:
    print(f"Path video: {path_video}")
    video_id = session.query(Video).filter(Video.path == str(path_video)).order_by(desc(Video.time)).first().id
    
    # add report
    report = Report(person_name, age, gender, mask, code_color, time, video_id, is_front)
    session.add(report)
    
    # report_id
    report_id = session.query(Report).filter(and_(
      Report.person_name == person_name, 
      Report.time == time
    )).first().id

    # add images for report
    for path in images_path:
      report_image = ReportImage(path, report_id)
      session.add(report_image)
  
  except Exception as e:
    session.rollback()

    raise Exception({ 
      "message": "Thêm report vào db không thành công !", 
      "error": f"Error {e}"
    })

  finally:  
    session.commit()
    session.close()

def get_reports_service(video_id):
  session = db_session()
  try:
    # params
    page_num = request.args.get("pageNum", type=int)
    page_size = request.args.get("pageSize", type=int)
    start_time = request.args.get("startTime", type=int)
    end_time = request.args.get("endTime", type=int)
    begin_age = request.args.get('beginAge', type=int)
    end_age = request.args.get('endAge', type=int)
    gender = request.args.get('gender', type=int)
    mask = request.args.get('mask', type=int)

    if start_time is None or end_time is None:
      return jsonify({"message": "Bad request !"}), 400
    
    else:
      reports = session.query(
        Report.id,
        Report.person_name,
        Report.age,
        Report.gender,
        Report.mask,
        Report.code_color,
        Report.time
      ).filter(and_(
        Report.video_id == video_id,
        Report.time >= start_time,
        Report.time <= end_time,
        Report.age >= begin_age,
        Report.age <= end_age,
        or_(
          Report.gender == gender,
          gender is None
        ),
        or_(
          Report.mask == mask,
          mask is None
        )
      ))

      totalReports = reports.count()
      
      # pagination
      if page_num is not None and page_size is not None:
        offset = (page_num - 1) * page_size
        reports = reports.order_by(desc(Report.id))\
          .offset(offset).limit(page_size).all()
      
      else:
        reports = reports.order_by(desc(Report.id)).all()
      
      for i, report in enumerate(reports):
        images = session.query(ReportImage).filter(ReportImage.report_id == report.id)
        reports[i] = report._asdict()
        reports[i]['images'] = images_schema.dump(images)

      return jsonify({
        "totalReports": totalReports,
        "reports": reports_schema.dump(reports)
      })
    
  except Exception as e:
    return jsonify({"message": f"Error {e}"}), 400
  
  finally:
    session.close()


def get_reports_db(video_id, page_num, page_size, start_time, end_time, begin_age, end_age, gender, mask, isface):
  session = db_session()
  try:
   
    reports = session.query(
      Report.id,
      Report.person_name,
      Report.age,
      Report.gender,
      Report.mask,
      Report.code_color,
      Report.time,
      Report.isface
    ).filter(and_(
      Report.video_id == video_id,
      Report.time >= start_time,
      or_(
        Report.time <= end_time,
        end_time == 0
      ),
      Report.age >= begin_age,
      Report.age <= end_age,
      or_(
        Report.gender == gender,
        gender is None
      ),
      or_(
        Report.mask == mask,
        mask is None
      ),
      or_(
        Report.isface == isface,
        isface is None
      )
    ))

    totalReports = reports.count()
    
    # pagination
    if page_num is not None and page_size is not None:
      offset = (page_num - 1) * page_size
      reports = reports.order_by(desc(Report.id))\
        .offset(offset).limit(page_size).all()
    
    else:
      reports = reports.order_by(desc(Report.id)).all()
    
    for i, report in enumerate(reports):
      images = session.query(ReportImage).filter(ReportImage.report_id == report.id)
      reports[i] = report._asdict()
      reports[i]['images'] = images_schema.dump(images)

    return reports
  except Exception as e:
    print(f"[get_reports_db]: {e}")
  
  finally:
    session.close()