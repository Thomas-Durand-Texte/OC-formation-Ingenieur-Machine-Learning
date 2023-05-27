(function($) {
$(function() {
  function bindNavigation() {
    $('#navigation').affix({
      offset: {
        top: 175
      }
    });
  }

  function bindRecentPosts() {
    $('.recent-posts').linkBuilding({
      bottom: 1500
    });
  }

  /**
   * Use `rel="external"` on your anchors to open them in a new window.
   */
  function bindExternalLinks() {
    $('a[rel*="external"]').attr('target', '_blank');
  }


  $(document).ready(function() {
    bindNavigation();
    bindRecentPosts();
    bindExternalLinks();
  });
});
}) (jQuery);

