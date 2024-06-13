<template>
  <div id="Content">
    <div id="Fire">
      <div id="Fire_image" style="display: flex; justify-content: normal;">
        <el-card
          id="Fire_image_1"
          class="box-card"
          style="
            border-radius: 8px;
            width: 1200px;
            height: 720px;
            background-color: rgba(255, 255, 255, 0.2);
          "
        >
          <div class="demo-image__preview1" >
            <div>
            <el-image
              :src="url_1"
              class="image_1"
              :preview-src-list="srcList"
              style="border-radius: 3px 3px 0 0; width: 400px; height: 600px; margin-left: 50px"
              @error="handleError"
            >
              <template #error>
                <div class="error">
                  <button
                    v-show="showbutton"
                    class="el-button el-button--primary download_bt"
                    @click="true_upload"
                    style="width: 200px; height: 40px; margin-top: 150px; font-size: 16px"
                  >
                    <i class="el-icon-upload"></i> 上传图像
                    <input
                      ref="upload"
                      style="display: none"
                      name="file"
                      type="file"
                      @change="update"
                    />
                  </button>
                </div>
              </template>
            </el-image>
            </div>
            <div class="img_info_1" style="border-radius: 0 0 5px 5px; width: 400px; height: 40px; margin-left: 50px; display: flex; align-items: center;">
              <span style="color: white; letter-spacing: 10px; font-size: 16px; margin: auto">原始图像</span>
            </div>
          </div>
          <div class="demo-image__preview2">
            <div
              v-loading.fullscreen.lock="loading"
            >
            <el-image
              :src="url_2"
              class="image_1"
              :preview-src-list="srcList1"
              style="border-radius: 3px 3px 0 0; width: 400px; height: 600px; margin-left: 200px; margin-right: 50px;"
              @error="handleError"
            >
              <template #error>
                <div class="error">{{ wait_return }}</div>
              </template>
            </el-image>
            </div>
            <div class="img_info_1" style="border-radius: 0 0 5px 5px; width: 400px; height: 40px; margin-left: 200px; display: flex; align-items: center;">
              <span style="color: white; letter-spacing: 10px; font-size: 16px; margin: auto">检测结果</span>
            </div>
          </div>
        </el-card>
      </div>
      <div id="info_fire">
        <el-card style="border-radius: 8px; width: 1200px; background-color: rgba(255, 255, 255, 0.2);">
          <div slot="header" class="clearfix" style="font-size: 30px; margin-bottom: 15px; font-weight: bold">
            <span >火点检测结果</span>
            <button
              style="margin-left: 40px; margin-bottom: 10px; width: 150px; height: 40px; font-size: 16px"
              v-show="!showbutton"
              class="el-button el-button--primary download_bt"
              @click="downloadImage"
            >
              <i class="el-icon-download"></i> 
                下载检测结果
              <input
                style="display: none"
                name="file"
                type="file"
              />
            </button>
            <button
              style="margin-left: 20px; margin-bottom: 10px; width: 150px; height: 40px; font-size: 16px"
              v-show="!showbutton"
              class="el-button el-button--primary download_bt"
              @click="true_upload2"
            >
              <i class="el-icon-upload"></i> 
                重新选择图像
              <input
                ref="upload2"
                style="display: none"
                name="file"
                type="file"
                @change="update"
              />
            </button>
          </div>
          <el-tabs v-model="activeName" type="border-card">
            <el-tab-pane label="火点分布情况" name="first">
              <div id="pie-chart" ref="pieChart"></div>
            </el-tab-pane>
            <el-tab-pane label="子图示例" name="second">
              <div class="subgraph-container">
                <label for="ksSelect" class="label">子图大小：</label>
                <el-select v-model="selectedKs" @change="updateSubGraph" id="ksSelect" class="select">
                  <!-- 添加选项 -->
                  <el-option label="15" value="15"></el-option>
                  <el-option label="25" value="25"></el-option>
                  <el-option label="75" value="75"></el-option>
                </el-select>
              </div>
              <el-table
                v-if="l_row !== null && r_row !== null && l_col !== null && r_col !== null"
                :data="coordinateData"
                height="100"
                border
                style="width: 400px; text-align: center; margin-left: 405px; margin-top: 20px; font-size: 20px; font-weight: bold;"
              >
                <el-table-column label="子图左上角坐标" prop="lCoordinate" width="200px"></el-table-column>
                <el-table-column label="子图右下角坐标" prop="rCoordinate" width="200px"></el-table-column>
              </el-table>
              <el-image
                :src="url_3"
                class="image_2"
                :preview-src-list="srcList2"
                @error="handleError"
              >
                <template #error>
                  <div class="error">{{ wait_return }}</div>
                </template>
              </el-image>
            </el-tab-pane>
            <el-tab-pane label="检测分数指标" name="third">
              <el-table
                :data="feature_list"
                height="100"
                border
                style="width: 700px; text-align: center; margin-left: 200px; margin-top: 20px; font-size: 20px; font-weight: bold;"
                :v-loading="loading" 
                element-loading-text="数据正在处理中，请耐心等待"
                element-loading-spinner="el-icon-loading"
                lazy>
                <el-table-column prop="tp" label="TP"  width="100px"></el-table-column>
                <el-table-column prop="fp" label="FP"  width="100px"></el-table-column>
                <el-table-column prop="tn" label="TN"  width="100px"></el-table-column>
                <el-table-column prop="fn" label="FN"  width="100px"></el-table-column>
                <el-table-column prop="p" label="准确率"  width="100px"></el-table-column>
                <el-table-column prop="r" label="召回率" width="100px"></el-table-column>
                <el-table-column prop="f1" label="F1值" width="100px"></el-table-column>
              </el-table>
            </el-tab-pane>
          </el-tabs>
        </el-card>
      </div>
    </div>
  </div>
</template>

<script>
import axios from "axios";
import * as echarts from 'echarts';

// const feat_list = ['tp', 'fp', 'tn', 'fn'];

  export default {
    // name: "Content",
    data() {
      return {
        server_url: "http://127.0.0.1:5003",
        activeName: "first",
        active: 0,
        centerDialogVisible: true,
        url_1: "",
        tmp_url_1:"",
        url_2: "",
        tmp_url_2:"",
        url_3:"",
        selectedKs: 75,
        download_url:"",
        textarea: "",
        srcList: [],
        srcList1: [],
        srcList2: [],
        feature_list: [],
        feature_list_1: [],
        l_row: null,
        r_row: null,
        l_col: null,
        r_col: null,
        // feat_list: [],
        // chart_list:[],
        url: "",
        visible: false,
        wait_return: "",
        wait_upload: "",
        loading: false,
        // loading_1: false,
        // loading_2: false,
        table: false,
        isNav: false,
        showbutton: true,
        percentage: 0,
        fullscreenLoading: false,
        opacitys: {
          opacity: 0,
        },
        dialogTableVisible: false,
      };
    },

    mounted() {
      this.initPieChart();
    },

    created: function () {
      document.title = "火点检测系统WEB端";
    },
    methods: {
      true_upload() {
        this.$refs.upload.click();
      },
      true_upload2() {
        this.$refs.upload2.click();
      },
      next() {
        this.active++;
      },
      // 获得目标文件
      getObjectURL(file) {
        var url = null;
        if (window.createObjcectURL != undefined) {
          url = window.createOjcectURL(file);
        } else if (window.URL != undefined) {
          url = window.URL.createObjectURL(file);
        } else if (window.webkitURL != undefined) {
          url = window.webkitURL.createObjectURL(file);
        }
        return url;
      },

      update(e) {
        this.url_1 = "";
        this.url_2 = "";
        this.url_3 = "";
        this.download_url="";
        this.srcList = [];
        this.srcList1 = [];
        this.srcList2 = [];
        this.wait_return = "";
        this.wait_upload = "";
        this.feature_list = [];
        this.feature_list_1= [];
        this.fullscreenLoading = true;
        this.loading = true;
        this.showbutton = false;
        this.tp = 0;
        this.fp = 0;

        let file = e.target.files[0];
        let ks = this.selectedKs
        this.url_1 = this.$options.methods.getObjectURL(file);
        let param = new FormData(); 
        param.append("file", file, file.name);
        param.append("ks", ks); 


        let config = {
          headers: { "Content-Type": "multipart/form-data" },
        }; 

        var randomTimeout = Math.random() * 5000 + 1000; // 1s-6s

        axios.post(this.server_url + "/upload", param, config)
          .then((response) => {

            this.url_1 = response.data.image_url;
            this.srcList.push(this.url_1);

            setTimeout(() => {
              this.url_2 = response.data.draw_url;
              this.download_url = response.data.draw_url;
              this.srcList1.push(this.url_2);
              this.notice1();
              this.loading = false;

              this.feature_list.push(response.data.image_info);
              this.feature_list_1 = Object.assign({}, response.data.image_info);

              let tp=this.feature_list_1.tp
              let fp=this.feature_list_1.fp
              let fn=this.feature_list_1.fn
              let tn=this.feature_list_1.tn
              let fire = tp+fp;
              let nofire = tn+fn;
              console.log(fire, nofire);
              this.initPieChart(tp, fp, fn);
            
            }, randomTimeout);
          })
          .catch((error) => {
            // 处理错误
            console.error("上传失败：", error);
            this.loading = false;
          });
      },

      updateSubGraph() {
        let ks = this.selectedKs
        let param = new FormData();
        param.append("ks", ks); 
        let config = {
          headers: { "Content-Type": "multipart/form-data" },
        }; 
        axios.post(this.server_url + "/upload_sub", param, config)
          .then((response) => {
            this.url_3 = response.data.sub_url;
            console.log(this.url_3);
            this.srcList2.push(this.url_3);
            
            this.l_row = response.data.l_row;
            this.r_row = response.data.r_row;
            this.l_col = response.data.l_col;
            this.r_col = response.data.r_col;
          })
          .catch((error) => {
            console.error("Error updating subgraph:", error);
          });
      },

      initPieChart(tp, fp, fn) {
        // 基于准备好的DOM，初始化ECharts实例
        this.chart = echarts.init(this.$refs.pieChart);

        // 配置选项
        let options = {
          tooltip: {
                    trigger: 'item',
                    formatter: '{b}: {c} ({d}%)'          
                  },
          legend: {
            data: [
              'TrueFire',
              'FalseFire',
              'TP',
              'FP',
              'FN'
            ]
          },
          series: [
            {
              name: '火点分布',
              type: 'pie',
              selectedMode: 'single',
              radius: [0, '30%'],
              label: {
                position: 'inner',
                fontSize: 10
              },
              labelLine: {
                show: false
              },
              data: [
                { value: tp, name: 'TrueFire', itemStyle: { color: '#e81e25' } },
                { value: fp, name: 'FalseFire', itemStyle: { color: '#fe7773' }}
              ]
            },
            {
              name: '火点分布',
              type: 'pie',
              radius: ['45%', '60%'],
              labelLine: {
                length: 30
              },
              label: {
                formatter: '{a|{a}}{abg|}\n{hr|}\n  {b|{b}:}{c}  {per|{d}%}  ',
                backgroundColor: '#F6F8FC',
                borderColor: '#8C8D8E',
                borderWidth: 1,
                borderRadius: 4,
                rich: {
                  a: {
                    color: '#6E7079',
                    lineHeight: 22,
                    align: 'center'
                  },
                  hr: {
                    borderColor: '#8C8D8E',
                    width: '100%',
                    borderWidth: 1,
                    height: 0
                  },
                  b: {
                    color: '#4C5058',
                    fontSize: 14,
                    fontWeight: 'bold',
                    lineHeight: 33
                  },
                  per: {
                    color: '#fff',
                    backgroundColor: '#4C5058',
                    padding: [3, 4],
                    borderRadius: 4
                  }
                }
              },
              data: [
                // { value: tn, name: 'TN' },
                { value: fn, name: 'FN',itemStyle: { color: '#f19f4d' } },
                { value: fp, name: 'FP',itemStyle: { color: '#d9d9d9'}},
                { value: tp, name: 'TP',itemStyle: { color: '#f9cf00'} }
              ]
            }
          ]
        };

        // 渲染图表
        this.chart.setOption(options);
      },

      downloadImage() {
        
        const imageUrl = this.download_url;

        // 发送 GET 请求获取图片数据
        fetch(imageUrl)
          .then(response => {
            if (!response.ok) {
              throw new Error('Network response was not ok');
            }
            return response.blob();
          })
          .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'fire_detection_results.png';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
          })
          .catch(error => {
            console.error('There has been a problem with your fetch operation:', error);
          });
      },

      myFunc() {
        if (this.percentage + 33 < 99) {
          this.percentage = this.percentage + 33;
        } else {
          this.percentage = 99;
        }
      },

      notice1() {
        this.$notify({
          title: "检测完毕",
          message: "点击图片可以查看大图",
          duration: 0,
          type: "success",
        });
      },
    },
    computed: {
      coordinateData() {
        if (this.l_row !== null && this.r_row !== null && this.l_col !== null && this.r_col !== null) {
          return [
            { lCoordinate: `(${this.l_row}, ${this.l_col})`, rCoordinate: `(${this.r_row}, ${this.r_col})` }
          ];
        } else {
          return [];
        }
      }
    }
  };
</script>

<style scoped>
/* * {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
} */

.dialog_info {
  margin: 20px auto;
}

.text {
  font-size: 14px;
}

.item {
  margin-bottom: 18px;
}

.clearfix:before,
.clearfix:after {
  display: table;
  content: "";
}

.clearfix:after {
  clear: both;
}

.box-card {
  width: 680px;
  height: 200px;
  border-radius: 8px;
  margin-top: -20px;
}

.divider {
  width: 50%;
}

#Fire {
  display: block;
  height: 100%;
  width: 100%;
  /* flex-wrap: wrap; */
  justify-content: center;
  margin: 0 auto;
  /* margin-right: 0px; */
  /* max-width: 1800px; */
  background-color: #0e0301;
}

#Fire_image_1 {
  width: 90%;
  height: 40%;
  margin: 0px auto;
  /* padding: 0px auto; */
  /* margin-right: 180px; */
  margin-bottom: 0px;
  border-radius: 4px;
}

/* #Fire_image {
  margin-bottom: 60px;
  margin-left: 30px;
  margin-top: 5px;
} */

.image_1 {
  width: 400px;
  height: 600px;
  background: #ffffff;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.image_2 {
  width: 406px;
  height: 406px;
  background: #ffffff;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  margin-left: 400px;
}

.img_info_1 {
  height: 30px;
  width: 275px;
  text-align: center;
  background-color: #f17773;
  line-height: 30px;
}

.demo-image__preview1 {
  width: 250px;
  height: 290px;
  margin: 20px 60px;
  float: left;
}

.demo-image__preview2 {
  width: 250px;
  height: 290px;

  margin: 20px 460px;
  /* background-color: green; */
}

.error {
  margin: 100px auto;
  width: 50%;
  padding: 10px;
  text-align: center;
}

.block-sidebar {
  position: fixed;
  display: none;
  left: 50%;
  margin-left: 600px;
  top: 350px;
  width: 60px;
  z-index: 99;
}

.block-sidebar .block-sidebar-item {
  font-size: 50px;
  color: lightblue;
  text-align: center;
  line-height: 50px;
  margin-bottom: 20px;
  cursor: pointer;
  display: block;
}

div {
  display: block;
}

.block-sidebar .block-sidebar-item:hover {
  color: #187aab;
}

.download_bt {
  padding: 10px 16px !important;
}

#upfile {
  width: 104px;
  height: 45px;
  background-color: #187aab;
  color: #fff;
  text-align: center;
  line-height: 45px;
  border-radius: 3px;
  box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.1), 0 2px 2px 0 rgba(0, 0, 0, 0.2);
  color: #fff;
  font-family: "Source Sans Pro", Verdana, sans-serif;
  font-size: 0.875rem;
}

.file {
  width: 200px;
  height: 130px;
  position: absolute;
  left: -20px;
  top: 0;
  z-index: 1;
  -moz-opacity: 0;
  -ms-opacity: 0;
  -webkit-opacity: 0;
  opacity: 0; /*css属性&mdash;&mdash;opcity不透明度，取值0-1*/
  filter: alpha(opacity=0);
  cursor: pointer;
}

#upload {
  position: relative;
  margin: 0px 0px;
}

.divider {
  background-color: #eaeaea !important;
  height: 2px !important;
  width: 100%;
  margin-bottom: 50px;
}

.divider_1 {
  background-color: #ffffff;
  height: 2px !important;
  width: 100%;
  margin-bottom: 20px;
  margin: 20px auto;
}

.steps {
  font-family: "lucida grande", "lucida sans unicode", lucida, helvetica,
    "Hiragino Sans GB", "Microsoft YaHei", "WenQuanYi Micro Hei", sans-serif;
  color: #fe7773;
  text-align: center;
  margin: 15px auto;
  font-size: 25px;
  font-weight: bold;
  text-align: center;
}

.step_1 {
  /*color: #303133 !important;*/
  margin: 20px 26px;
}

#info_fire {
  margin-top: 30px;
  display: flex;
  justify-content: center;
}

#pie-chart {
  width: 100%;
  height: 380px;
}

.custom-table .el-table__body-wrapper {
  text-align: center; /* 居中显示 */
}

.custom-table .el-table__body td div {
  font-size: 16px; /* 放大字体 */
  font-weight: bold; /* 加粗字体 */
}

.subgraph-container {
  display: flex;
  align-items: center;
}

.label {
  margin-right: 10px; /* 调整标签与下拉框之间的间距 */
  font-size: 20px;
  font-weight: bold;
  color: #0e0301
}

.select {
  width: 100px;
}
</style>
