# -*- coding:utf-8 -*-
import os
import json
import requests
import flask
from gevent import pywsgi
if __name__ == '__main__':
    write_dir = r'F:\PycharmProject\kbqa\CBLUE-main\CaseDatasets\KUAKE-QIC'
    res_dir = r'F:\PycharmProject\kbqa\CaseDataSet\outputresult'

    app = flask.Flask(__name__)

    @app.route("/service/api/case",methods=["GET","POST"])
    def medical_qa():
        data = {"sucess":0}
        param = flask.request.get_json()
        text = param["text"]

        # 1.把接口输入写入到文件，封装成数据集
        # 如果存在test文件则删除，不存在则创建
        def queryToFile(query):
            if os.path.exists(write_dir + '\KUAKE-QIC_test.json'):
                os.remove(write_dir + '\KUAKE-QIC_test.json')
            with open(os.path.join(write_dir, 'KUAKE-QIC_test.json'), mode='w', encoding='utf-8',
                      newline='') as out_file:
                # 读取所有数据到data
                data = []
                item = {}
                item['id'] = 's1'
                item['query'] = query
                data.append(item)
                json.dump(data, out_file, ensure_ascii=False, indent=2)
                out_file.close()
            return data
        queryToFile(text)

        # 2.运行模型预测结果保存到文件
        Cmd = "bash examples/run_qicCase.sh predict"
        output = os.popen(Cmd)
        print(output.read())
        # print('\n******************自动派单完成*******************\n')

        # 3.读取结果文件
        def read_RES(readfile):
            with open(os.path.join(res_dir, readfile), 'rt', encoding="utf-8") as readfile:
                data = json.load(readfile)
            return data[0]
        res = read_RES('KUAKE-QIC_test.json')
        # print(res['label'])
        print(res)

        data["query"] = text
        data["labels"] = res['labels']
        data["sucess"] = 1
        return flask.jsonify(data)

    server = pywsgi.WSGIServer(("0.0.0.0",60000), app)
    # print('\n******************自动派单服务已启动*******************\n')
    server.serve_forever()



