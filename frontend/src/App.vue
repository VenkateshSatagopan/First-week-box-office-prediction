<template>
  <div id="app">

    <router-link to="/">Home</router-link> |
    <router-link to="/api_example">API example</router-link>

    <router-view/>

  </div>
</template>



<script>
import Home from './views/Home.vue'

export default{
  name:'app',
  components:{
    Home
  }
}
</script>



<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #282828;
  margin-top: 60px;
}

</style>
