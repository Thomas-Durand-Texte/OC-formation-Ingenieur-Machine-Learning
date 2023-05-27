;(function ($, window, document, undefined) {

    var pluginName = "linkBuilding",
        dataKey = "plugin_" + pluginName;

    var Plugin = function (element, options) {

        this.element = element;

        this.options = {
            bottom: 300
        };

        this.init(options);
    };

    Plugin.prototype = {
        init: function (options) {
            $.extend(this.options, options);
            that = this;

            var doc = document.documentElement, body = document.body;
            var height_screen = (doc && doc.offsetHeight  || body && body.offsetHeight  || 0);
            var height_article = $('.entry-content').height();
            var bottom = this.options.bottom;

            var scrollCount = 0;
            var didScroll = false;

            window.onscroll = launch;

            function launch () {
              didScroll = true;
            }

            function engine () {

              setInterval(function() {
                if(didScroll) {
                  didScroll = false;

                  // delay scroll event with counter
                  scrollCount++;

                  var scrollBottom = $(window).scrollTop() + $(window).height();
                  bottom_position = height_screen - scrollBottom;

                  if (scrollCount === 2) {
                    if (bottom_position < bottom && bottom_position > 300) {
                      that.element.addClass('open');
                    } else {
                     that.element.removeClass('open');
                    }

                  }
                }

                //reset count for event scroll
                if (didScroll === false && scrollCount > 1) {
                  scrollCount = 0;
                }

              }, 50);
            }

            engine();



        },

    };

    /*
     * Plugin wrapper, preventing against multiple instantiations and
     * return plugin instance.
     */
    $.fn[pluginName] = function (options) {

        var plugin = this.data(dataKey);

        // has plugin instantiated ?
        if (plugin instanceof Plugin) {
            // if have options arguments, call plugin.init() again
            if (typeof options !== 'undefined') {
                plugin.init(options);
            }
        } else {
            plugin = new Plugin(this, options);
            this.data(dataKey, plugin);
        }

        return plugin;
    };

}(jQuery, window, document));

