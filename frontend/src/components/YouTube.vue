<template>
  <div>

    <iframe width="560" height="315" src="https://www.youtube.com/watch?v=TcMBFSGVi1c" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
      <p>Name of the trailer</p>
    </iframe>

    <ul class="list-unstyled">
    <b-media tag="li">
      <b-img slot="aside" blank blank-color="#abc" width="64" alt="placeholder"></b-img>

      <h5 class="mt-0 mb-1">Actor Name 1</h5>
      <p class="mb-0">
        Cras sit amet nibh libero, in gravida nulla. Nulla vel metus scelerisque ante sollicitudin.
      </p>
    </b-media>

    <b-media tag="li" class="my-4">
      <b-img slot="aside" blank blank-color="#cba" width="64" alt="placeholder"></b-img>

      <h5 class="mt-0 mb-1">Actor Name 2</h5>
      <p class="mb-0">
        Cras sit amet nibh libero, in gravida nulla. Nulla vel metus scelerisque ante sollicitudin.
      </p>
    </b-media>

    <b-media tag="li">
      <b-img slot="aside" blank blank-color="#bac" width="64" alt="placeholder"></b-img>

      <h5 class="mt-0 mb-1">Actor Name 3</h5>
      <p class="mb-0">
        Cras sit amet nibh libero, in gravida nulla. Nulla vel metus scelerisque ante sollicitudin.
      </p>
    </b-media>
  </ul>

  </div>
</template>



<script>
export default {
  name: 'YouTube',
}
</script>



<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>


</style>
