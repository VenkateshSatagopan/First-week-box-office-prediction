<template>
  <div id="Home">
    <MainPage msg = 'hello mainpage!'></MainPage>
    <MainSearch></MainSearch>
    <MetaData msg = 'hello metadata!'></MetaData>
    <Result></Result>
    <YouTube></YouTube>
  </div>
</template>



<script>
// @ is an alias to /src
import MainPage from '@/components/MainPage.vue'
import MainSearch from '@/components/MainSearch.vue'
import MetaData from '@/components/MetaData.vue'
import Result from '@/components/Result.vue'
import YouTube from '@/components/YouTube.vue'

export default {
  name: 'Home',
  components: {
    MainPage,
    MainSearch,
    MetaData,
    Result,
    YouTube
  }
}
</script>
