<template>
  <div class="about">
    <h1>This is an example for API calls</h1>
    You need to run the backend server to get responses here.

    <b-form>
    <label class="sr-only" for="inline-form">Name</label>
    <b-input
      id="inline-form-input-name"
      class="mb-2 mr-sm-2 mb-sm-0"
      placeholder="API call"
      v-model="API_request"
    ></b-input>

    <b-button variant="primary" @click='getFromBackend'>get</b-button>
  </b-form>
  <p><span v-html="API_response"></span></p>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'APIexample',
  components: {
  },
data() {
  return {
    API_request: '',
    API_response: ''
  }
},
methods: {

  getFromBackend () {
    const path = 'http://'+location.hostname+':80/api/'
    axios.get(path+this.API_request)
    .then(response => {
      this.API_response = response
    })
    .catch(error => {
      this.API_response = '<p style="color:red;">API is unreachable</p>'
      console.log(error)
    })
  }
}
}
</script>
