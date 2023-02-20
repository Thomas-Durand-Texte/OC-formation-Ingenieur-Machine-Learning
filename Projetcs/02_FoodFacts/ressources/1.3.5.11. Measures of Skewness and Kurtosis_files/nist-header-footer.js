/**
 * @file
 * Header and footer scripts
 *
 */

$(document).ready(function () {

  $("body").prepend('<div id="nistheadergoeshere"></div>');
  $.ajax({
    url: "https://www.itl.nist.gov/div898/handbook/nist-header-footer/boilerplate-header.html",
    cache: false,
    dataType: "html",
    success: function (data) { $('#nistheadergoeshere').append(data); },
  });

  $("body").append('<div id="nistfootergoeshere"></div>');
  $.ajax({
    url: "https://www.itl.nist.gov/div898/handbook/nist-header-footer/boilerplate-footer.html",
    cache: false,
    dataType: "html",
    success: function (data) { $('#nistfootergoeshere').append(data); },
  });

});
