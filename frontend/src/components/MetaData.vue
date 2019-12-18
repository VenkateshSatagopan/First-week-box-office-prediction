<template>
  <div>
    <h3> Movie's Name input </h3>
    <h4>Rating: </h4>
    <h4>Genre: </h4>
    <h4>Directy By: </h4>
    <h4>Written By: </h4>
    <h4>In Theaters: </h4>
    <h4>Runtime: </h4>
    <h4>Studio: </h4>

  </div>
</template>



<script>
export default {
  name: 'MetaData',
  props: {
    msg: String
  }
}
</script>



<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h3 {
  margin: 24px 100px 0;
  width: 264px;
  height: 32px;
  object-fit: contain;
  font-family: Lato;
  font-size: 20px;
  font-weight: 900;
  font-style: normal;
  font-stretch: normal;
  line-height: normal;
  letter-spacing: 0.83px;
  color: #212121;
}

h4 {
  margin: 16px 50px 0;
  width: 300px;
  height: 32px;
  object-fit: contain;
  font-family: Roboto;
  font-size: 16px;
  font-weight: 500;
  font-style: normal;
  font-stretch: normal;
  line-height: 2.25;
  letter-spacing: normal;
  color: #4a4a4a;
}


</style>
