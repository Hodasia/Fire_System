import datetime
import logging as rel_log
import os
from datetime import timedelta
from flask import *
from processor.fire_utils import save_results, show_sub, fire_detect, print_img

UPLOAD_FOLDER = r'./tmp/ct'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'npz'])
app = Flask(__name__)
app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

# 添加类保留变量值
class DataBase(object):
    sub_img_pid = None
    cnt = 0

data = DataBase()

# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    return redirect(url_for('static', filename='./index.html'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    file = request.files['file']
    print(datetime.datetime.now(), file.filename)
    if file and allowed_file(file.filename):
        pid = file.filename
        data.sub_img_pid = pid

        # Version 1: 直接跑结果
        # origin_img, results_img, image_info = fire_detect(file, pid.rsplit('.', 1)[0])
        
        # origin_path = os.path.join('./tmp/ct', pid.rsplit('.', 1)[0]+'.png')
        # print_img(origin_img, origin_path, 'origin')

        # results_path = os.path.join('./tmp/draw', pid.rsplit('.', 1)[0]+'.png')
        # print_img(results_img, results_path, 'results')
        ##########

        # Version 2: 直接显示本地结果
        image_info = save_results(pid)
        ##########

        return jsonify({'status': 1,
                        'image_url': 'http://127.0.0.1:5003/tmp/ct/' + pid,
                        'draw_url': 'http://127.0.0.1:5003/tmp/draw/' + pid,
                        'image_info': image_info})

    return jsonify({'status': 0})

# 上传子图
@app.route('/upload_sub', methods=['GET', 'POST'])
def upload_sub_file():
    selected_ks = int(request.form['ks'])
    sub_img_pid = data.sub_img_pid
    data.cnt += 1
    print(sub_img_pid, selected_ks)
    if sub_img_pid:
        sub_pid = sub_img_pid.rsplit('.', 1)[0]
        sub_img_pid = sub_pid + '_'+ str(data.cnt) + '.png'
        l_row, r_row,l_col, r_col = show_sub(sub_pid, selected_ks, data.cnt)
        return jsonify({'sub_url': 'http://127.0.0.1:5003/tmp/sub/' + sub_img_pid,
                       'l_row':l_row, 'r_row':r_row, 'l_col':l_col, 'r_col':r_col,
                       'status': 1})
    else:
        return jsonify({'status': 0})


@app.route("/download", methods=['GET'])
def download_file():
    file_url = request.args.get('file_url')
    if file_url:
        file_name = file_url.split('/')[-1]  # 从 URL 中获取文件名
        download_path = os.path.join('./tmp/download', file_name)
        if os.path.exists(download_path):
            return send_from_directory(download_path, as_attachment=True)
        else:
            return jsonify({'status': 0, 'message': 'File not found'}), 404
    else:
        return jsonify({'status': 0, 'message': 'File URL not provided'}), 400


# show photo
@app.route('/tmp/<path:file>', methods=['GET'])
def show_photo(file):
    if request.method == 'GET':
        if not file is None:
            path, filename = file.rsplit('/', 1)
            # 提取文件名中的数字部分
            file_number = filename.split('.')[0]
            # 构造新的文件名
            new_filename = file_number + '.png'
            # 构造新的文件路径字符串
            new_file = path + '/' + new_filename
            image_data = open(f'tmp/{new_file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response


if __name__ == '__main__':
    files = [
        'tmp/ct', 'tmp/draw', 'tmp/sub'
    ]
    for ff in files:
        if not os.path.exists(ff):
            os.makedirs(ff)
    app.run(host='127.0.0.1', port=5003, debug=True)
