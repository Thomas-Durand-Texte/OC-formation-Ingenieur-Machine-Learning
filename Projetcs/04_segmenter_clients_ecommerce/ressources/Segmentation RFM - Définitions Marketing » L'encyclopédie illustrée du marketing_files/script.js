(function($) {

	"use strict";


    /*------------------------------------------
        = FUNCTIONS
    -------------------------------------------*/

    // Toggle mobile navigation
    function toggleMobileNavigation() {
        var navbar = document.getElementById('menu');
        var openBtn = document.getElementById('open-btn');
        var closeBtn = document.getElementById('close-navbar');

        openBtn.addEventListener("click", function(){
            navbar.style.display="block";
            openBtn.style.display="none";
            closeBtn.style.display="block";
        }, false);

        closeBtn.addEventListener("click", function(){
            navbar.style.display="none";
            openBtn.style.display="block";
            closeBtn.style.display="none";
        }, false);
    }

    toggleMobileNavigation();


    // Function for toggle a class for small menu
    function toggleClassForSmallNav() {
        var windowWidth = window.innerWidth;
        var mainNav = $("#navbar > ul");

        if (windowWidth <= 991) {
            mainNav.addClass("small-nav");
        } else {
            mainNav.removeClass("small-nav");
        }
    }

    toggleClassForSmallNav();


    // Function for small menu
    function smallNavFunctionality() {
        var windowWidth = window.innerWidth;
        var mainNav = $(".navigation-holder");
        var smallNav = $(".navigation-holder > .small-nav");
        var subMenu = smallNav.find(".sub-menu");
        var megamenu = smallNav.find(".mega-menu");
        var menuItemWidthSubMenu = smallNav.find(".menu-item-has-children > a");

        if (windowWidth <= 991) {
            subMenu.hide();
            megamenu.hide();
            menuItemWidthSubMenu.on("click", function(e) {
                var $this = $(this);
                $this.siblings().slideToggle();
                 e.preventDefault();
                e.stopImmediatePropagation();
            })
        } else if (windowWidth > 991) {
            mainNav.find(".sub-menu").show();
            mainNav.find(".mega-menu").show();
        }
    }

    smallNavFunctionality();

    /*------------------------------------------
        = STICKY HEADER
    -------------------------------------------*/

    var lastScrollTop = '';

    /* function stickyMenu($targetMenu, $toggleClass) {
        var st = $(window).scrollTop();
        var mainMenuTop = $('.site-header .navigation');
        var windowWidth = window.innerWidth;
        
        if (windowWidth > 991) {
            if ($(window).scrollTop() > 200) {
                $targetMenu.addClass($toggleClass);
            } else {
                $targetMenu.removeClass($toggleClass);
            }
        }
    } */
    
    function stickySearch() {
        var st = document.body.scrollTop || document.documentElement.scrollTop;
        var nav = document.getElementById('nav-fixed');
        var bar = document.getElementById('fixed-search');
        var bouton = document.getElementById('fixed-bouton-search');
        var windowWidth = window.innerWidth;
        
        if (windowWidth > 991) {
            if (st > 200) {
                nav.style.display="block";
                window.setTimeout( function(){
                    nav.style.top="0";
                    bar.style.top="0";
                    bouton.style.top="0";
                },100);
            } else {
                nav.style.top="-100px";
                bar.style.top="-100px";
                bouton.style.top="-100px";
                window.setTimeout( function(){
                    nav.style.display="none";
                },500);
            }
        }
    } 
    
    /*------------------------------------------
        = BACK TO TOP BTN SETTING
    -------------------------------------------*/
    $("body").append("<a href='#' class='back-to-top'><i class='fa fa-angle-up'></i></a>");

    function toggleBackToTopBtn() {
        var amountScrolled = 300;
        if ($(window).scrollTop() > amountScrolled) {
            $("a.back-to-top").fadeIn("slow");
        } else {
            $("a.back-to-top").fadeOut("slow");
        }
    }

    $(".back-to-top").on("click", function() {
        $("html,body").animate({
            scrollTop: 0
        }, 700);
        return false;
    })
    
    /* document.getElementById('open-search').addEventListener("click", function(){
        document.getElementById('search').style.opacity="1";
        document.getElementById('search').style.margin="0";
        document.getElementById('close-search').style.right="15px";
        document.getElementById('close-search').style.opacity="1";
        document.getElementById('position_menu').style.zIndex="1";
    }, false);
    
    document.getElementById('close-search').addEventListener("click", function(){
        document.getElementById('search').style.opacity="0";
        document.getElementById('search').style.margin="0 -100vw";
        document.getElementById('close-search').style.right="100vw";
        document.getElementById('close-search').style.opacity="0";
        window.setTimeout(function(){
            document.getElementById('position_menu').style.zIndex="999";
        },300);
    }, false); */



    /*==========================================================================
        WHEN DOCUMENT LOADING
    ==========================================================================*/
        $(window).on('load', function() {

            toggleMobileNavigation();

            smallNavFunctionality();

        });



    /*==========================================================================
        WHEN WINDOW SCROLL
    ==========================================================================*/
    $(window).on("scroll", function() {

        /* if ($(".site-header").length) {
            stickyMenu( $('.site-header .navigation'), "menu-sticky" );
        } */
        
        stickySearch();

        toggleBackToTopBtn();

    });


    /*==========================================================================
        WHEN WINDOW RESIZE
    ==========================================================================*/
    $(window).on("resize", function() {

        toggleClassForSmallNav();

        clearTimeout($.data(this, 'resizeTimer'));

        $.data(this, 'resizeTimer', setTimeout(function() {
            smallNavFunctionality();
        }, 200));

    });

})(window.jQuery);
