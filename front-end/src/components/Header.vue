<template>
  <div id="Header">
    <!-- <div id="word">
      <h1>{{ msg }}</h1>
    </div> -->
    <!-- <h1>{{ msg }}</h1> -->
    {{ msg }}
  </div>
</template>
<script>
export default {
  name: "Header",
  data() {
    return {
      msg: "基于MODIS卫星数据的火点检测系统",
      activeIndex: "1",
    };
  },
  methods: {},
};
</script>
<style scoped>
#Header {
  /* padding: 0 0 10px 0; */
  width: 100%;
  margin: 0 0;
  background-color: #0e0301;
  text-align: center;
  color: #ffc2c3;
  letter-spacing: 10px;
  font-size: 3em;
  font-family: '华文细黑', Georgia;
  font-weight: bold;
}

/* #word {
  margin-left: 50%;
  margin-top: -35px;
  margin-bottom: 37px;
  height: 60px;
  line-height: 3.2em;
  letter-spacing: 8px;
} */

h1 {
  text-align: center;
  color: #21b3b9;
  letter-spacing: 20px;
  font-size: 3em;
}

.el-menu-demo {
  width: 80%;
  margin: 0px auto;
  padding: 0px auto;
}

.top-left-edition span i {
  float: left;
  margin-right: 10px;
}

i,
input,
label {
  vertical-align: middle;
}

i {
  border: 0;
  display: block;
  cursor: pointer;
}

.top-left-edition span {
  float: left;
  font-size: 16px;
  color: #999999;
  line-height: 24px;
  margin-right: 40px;
}
</style>


