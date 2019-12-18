<template>
  <div>

    <h1>Boxoffice Teller</h1>
    <h2>AI Boxoffice Prediction</h2>
    <h3> - with just a trailer</h3>

    <!-- Search Bar -->
    <!-- <div class="input-group">
      <input type="text" class="form-control" aria-label="Text input with dropdown button">
      <div class="input-group-append">
        <button class="btn btn-outline-secondary dropdown-toggle" type="button" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">Search</button>
        <div class="dropdown-menu">
          <a class="dropdown-item" href="#">Action</a>
          <a class="dropdown-item" href="#">Another action</a>
          <a class="dropdown-item" href="#">Something else here</a>
          <div role="separator" class="dropdown-divider"></div>
          <a class="dropdown-item" href="#">Separated link</a>
        </div>
      </div>
    </div>
  </div> -->
  <b-form-input v-model="text" placeholder="Insert IMDb ID"></b-form-input>
  <div class="mt-2">Value: {{ text }}</div>

  <b-form-input v-model="text" placeholder="Insert YouTube URL"></b-form-input>
  <div class="mt-2">Value: {{ text }}</div>

  <b-dropdown id="dropdown-1" text="Search" class="m-md-2">
    <b-dropdown-item>Search IMDb ID</b-dropdown-item>
    <b-dropdown-item>Search YouTube</b-dropdown-item>
    <b-dropdown-item>Search Both</b-dropdown-item>
  </b-dropdown>
</div>
</template>

<script>
export default {
  name: 'MainSearch',

  data (){
    return {
      text: ''
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h1 {
  margin-left:72px;
  width: 300px;
  height: 32px;
  object-fit: contain;
  font-family: Roboto;
  font-size: 24px;
  font-weight: bold;
  font-style: normal;
  font-stretch: normal;
  line-height: normal;
  letter-spacing: normal;
  color: #aeaeae;
}

h2 {
  margin: 88px 100px 0;
  width: 585px;
  height: 72px;
  object-fit: contain;
  font-family: Roboto;
  font-size: 50px;
  font-weight: 900;
  font-style: normal;
  font-stretch: normal;
  line-height: normal;
  letter-spacing: normal;
  color: #aeaeae;
}

h3 {
  margin: 24px 100px 0;
  width: 412px;
  height: 72px;
  object-fit: contain;
  font-family: Roboto condensed;
  font-size: 50px;
  font-weight: normal;
  font-style: normal;
  font-stretch: condensed;
  line-height: normal;
  letter-spacing: normal;
  color: #aeaeae;
}

body {
  background-color: #282828;
}

b-form-input {
  margin: 24px 100px 0;
  width: 300px;


}

</style>
