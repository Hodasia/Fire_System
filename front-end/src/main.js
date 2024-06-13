import './assets/main.css'

// import { createApp } from 'vue'
// import App from './App.vue'

// createApp(App).mount('#app')

import { createApp } from 'vue';
import { createRouter, createWebHistory } from 'vue-router';
import App from './App.vue';
import axios from 'axios';
import ElementPlus from 'element-plus';
import '../node_modules/element-plus/theme-chalk/index.css';
import '../src/assets/style.css';
import './theme/index.css';
import * as echarts from 'echarts';

const app = createApp(App);

app.config.productionTip = false;

app.use(ElementPlus);
app.use(axios);

app.config.globalProperties.$echarts = echarts;

const routes = [
  { path: '/App', component: App, meta: { title: '火点检测系统' } }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

app.use(router);

app.mount('#app');
