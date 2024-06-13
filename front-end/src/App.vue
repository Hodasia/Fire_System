<template>
  <div id="app">
    <app-header></app-header>
    <app-content></app-content>
  </div>
</template>

<script>
import Header from "./components/Header.vue";
import Content from "./components/Content.vue";
export default {
  name: "火点检测系统",
  data() {
    return {};
  },
  components: {
    "app-header": Header,
    "app-content": Content,
  },
  methods: {
  },
};

</script>

<style scoped>
</style>

