$=jQuery.noConflict();if(window.location.hash){window.scrollTo(0,0);setTimeout(function(){window.scrollTo(0,0)},1)}
function faq_slide(el){event.preventDefault();$(el).parent().toggleClass('expanded');$(el).parent().next(".qa-faq-answer").toggleClass('expanded');$(el).parent().next(".qa-faq-answer").slideToggle(400,function(){if(typeof tableOfContents!=="undefined"){tableOfContents()}})}
function scribbrPlayPauseVideo(videoClass){let videos=document.querySelectorAll(videoClass);videos.forEach((video)=>{video.muted=!0;let playPromise=video.play();if(playPromise!==undefined){playPromise.then((_)=>{let observer=new IntersectionObserver((entries)=>{entries.forEach((entry)=>{if(entry.intersectionRatio!==1&&!video.paused){video.pause()}else if(video.paused){video.play()}})},{threshold:0.2});observer.observe(video)})}})}
function isInViewport(el){var elementTop=el.offset().top;var elementBottom=elementTop+el.outerHeight();var viewportTop=$(window).scrollTop();var viewportBottom=viewportTop+$(window).height();return elementBottom>viewportTop&&elementTop<viewportBottom};function scribbr_checklist_students(el){var originalHeights=[];$(el).find('label').hover(function(){$(this).parent('li.checklist__list__item').toggleClass('hover')});$(el).find('input[type="checkbox"]').change(function(){var checkboxes=$(this).parents('ul.checklist__list').find('input[type="checkbox"]');var checked_checkboxes=$(this).parents('ul.checklist__list').find('input[type="checkbox"]:checked');$(this).parents('li.checklist__list__item').toggleClass('checked');$(this).parents('li.checklist__list__item').toggleClass('unchecked');$(this).parents('.scribbr-checklist').find('.count-checked').html(checked_checkboxes.length);if(checkboxes.length==checked_checkboxes.length&&$(this).parents('.scribbr-checklist').find('.scribbr-checklist__congratulations').length){var listelement=$(this).parents('.scribbr-checklist');originalHeights[listelement.attr('id')]=listelement.css('height');console.log(originalHeights);$(this).parents('.scribbr-checklist').animate({height:"750px"},200,function(){$(this).find('.scribbr-checklist__congratulations').fadeIn()})}});$(el).find('.close-congratulations').click(function(event){event.preventDefault();$(this).parents('.scribbr-checklist').find('.scribbr-checklist__congratulations').fadeOut(function(){$(this).parents('.scribbr-checklist').animate({height:originalHeights[$(this).parents('.scribbr-checklist').attr('id')]},200)})})}
$(document).ready(function(){$("body").append('<div id="scribbr-tooltip"></div>');$("#scribbr-tooltip").hide();$(".scribbr-tooltip").hover(function(){var tooltipContent=$(this).children(".tooltip-content").html();$("#scribbr-tooltip").html(tooltipContent);var top=$(this).offset().top-30;var left=$(this).offset().left-380;if(left<20){left=20}
$("#scribbr-tooltip").css({position:'absolute',top:top+'px',left:left+'px'});$("#scribbr-tooltip").stop(!0,!0).fadeIn()},function(){$("#scribbr-tooltip").stop(!0,!0).fadeOut()});$("#scribbr-tooltip").hover(function(){$("#scribbr-tooltip").stop(!0,!0).fadeIn()},function(){$("#scribbr-tooltip").stop(!0,!0).fadeOut()});scribbr_checklist_students('.scribbr-checklist--students');$('.checklists-slidedown .scribbr-checklist .checklist__header').click(function(){$(this).toggleClass('expanded');$(this).next().slideToggle()});$('.kb-collapse .collapse-header').click(function(){$(this).parents('.kb-collapse ').toggleClass('expanded').find('.collapse-content').slideToggle()});$(".qa-faq-anchor").click(function(){scribbr_add_event_to_ga('FAQ',document.location.pathname,$(this).text())});$("a.qa-faq-anchor").click(function(event){faq_slide(this)});$(".scribbrTabs-menu a").click(function(event){if(!$(this).parent().hasClass("current")){var index=$(this).parent().index();event.preventDefault();$(this).parent().addClass("current");$(this).parent().siblings().removeClass("current");$(this).closest(".scribbrTabs").find(".scribbrTabs-content").css("display","none");$(this).closest(".scribbrTabs").find(".scribbrTabs-content:nth-of-type("+(index+1)+")").fadeIn()}
return!1});controlUserInfoButtons();controlLanguageSelector();var subNavExists=document.getElementsByClassName('sub-nav'),isCitationPage=document.body.classList.contains('page-template-citation-generator');if(subNavExists.length>0){makeStickyElement('.sub-nav')}else if(!isCitationPage){makeStickyElement('.main-nav')}
smoothScroll();$('.radioWrap input[type="radio"],.checkboxWrap input[type="checkbox"]').each(function(index){if($(this).is(":checked")){$(this).closest('.radioWrap,.radioDropdownItem,.checkboxWrap').addClass('checked')}else{$(this).closest('.radioWrap,.radioDropdownItem.checkboxWrap').removeClass('checked')}});$('.radioWrap input[type="radio"],.checkboxWrap input[type="checkbox"]').change(function(){$('.radioWrap input[type="radio"],.checkboxWrap input[type="checkbox"]').each(function(index){if($(this).is(":checked")){$(this).closest('.radioWrap,.radioDropdownItem,.checkboxWrap').addClass('checked')}else{$(this).closest('.radioWrap,.radioDropdownItem,.checkboxWrap').removeClass('checked')}})})
$('.cta-services.style-horizontal').each(function(){var highestBox=0;$(this).find('.service-box__body').each(function(){if($(this).height()>highestBox){highestBox=$(this).height()}});if(highestBox>0){$(this).find('.service-box__body').height(highestBox)}})});function resizeOneThirdContainer(){var maxHeight=0;setTimeout(function(){if($(".oneThirdContainer .content p").length>0){$(".oneThirdContainer .content p").each(function(){if(maxHeight<$(this).height()){maxHeight=$(this).height()}});$(".oneThirdContainer .content").height(maxHeight)}},200)}
function controlLanguageSelector(){$("#languag").mouseover(function(){var closeLanguageSelector;var langSelector=$('#languageSelector');langSelector.fadeIn(200).hover(function(){clearTimeout(closeLanguageSelector)},function(){closeLanguageSelector=setTimeout(function(){$("#languageSelector").fadeOut()},500)});$(document).mouseup(function(e){var container=$("#languageSelector");if(!container.is(e.target)&&container.has(e.target).length===0){container.fadeOut()}})})}
function getRootDomain(){var fullDomain=document.location.host;var subdomain=fullDomain.split(".")[0];var rootDomain=fullDomain.split(subdomain)[1];return rootDomain}
function preload(arrayOfImages){$(arrayOfImages).each(function(){$('<img/>')[0].src=this})}
function analyticsActive(){var active;if(typeof ga=="undefined"){active=!1}else{active=!0}
return active}
function getBackendMessages(service,language){var jsonUrl="/wp-content/themes/scribbr-2017/scribbr-message/";$.getJSON(jsonUrl,function(data){processBackendMessages(data,service,language)})}
function processBackendMessages(json,service,language){var messageItem=service+"-message-"+language;var messageClassItem=service+"-message-class-"+language;var messageSelector=$("#backendJsonMessage");if(json[messageItem]!=null&&json[messageItem].trim()!=""){messageSelector.addClass(json[messageClassItem]).html("<p>"+json[messageItem]+"</p>").slideDown('fast')}}
function getBackendMessages_2017(service,language){var jsonUrl="/wp-content/themes/scribbr-2017/scribbr-message/";$.getJSON(jsonUrl,function(data){processBackendMessages_2017(data,service,language)})}
function processBackendMessages_2017(json,service,language){var messageKey=service+"-message-"+language;var classKey=service+"-message-class-"+language;var announcementMessage=json[messageKey];var announcementClass='announcement-banner--'+json[classKey];var announcementBanner=$(".announcement-banner");if(announcementMessage!=null&&announcementMessage.trim()!=""){announcementBanner.addClass(announcementClass).html("<div class='announcement-banner__close'><i class='fa fa-close'></i></div>"+"<div class='announcement-banner__message'>"+announcementMessage+"</div>")}
$('.announcement-banner__close').on("click",function(){$('.announcement-banner').hide()})}
function scribbr_add_event_to_ga(cat,act,lab,nonInteraction){act=typeof act!=='undefined'?act:'';nonInteraction=typeof nonInteraction!=='undefined'?nonInteraction:!1;dataLayer.push({'event':'GAEvent','eventCategory':cat,'eventAction':act,'eventLabel':lab,'eventNonInteraction':nonInteraction})}
function getSfRole(cookieName){var roles=Cookies.getJSON(cookieName);var role;if($.inArray("ROLE_ADMIN",roles)>-1){role="ROLE_ADMIN"}else if($.inArray("ROLE_SUPPORT",roles)>-1){role="ROLE_ADMIN"}else if($.inArray("ROLE_EDITOR",roles)>-1){role="ROLE_EDITOR"}else if($.inArray("ANONYMOUS",roles)>-1){role="ANONYMOUS"}else{role="ROLE_USER"}
return role}
function controlUserInfoButtons(){var symfonyEnv;if(ENV=="DEV"){symfonyEnv="TRANS"}else{symfonyEnv=ENV}
var cookieName="sf_"+symfonyEnv.toLowerCase()+"_roles";var role;if(typeof Cookies.get(cookieName)!=="undefined"){role=getSfRole(cookieName)}else{$('.js-accountButtons').append('<img src="'+rootUrl+'/pixel.gif" alt="pixel" id="setCookie" width="1" height="1">');$("img#setCookie").one('load',function(){if(typeof Cookies.get(cookieName)!=="undefined"){role=getSfRole(cookieName)}else{role="ANONYMOUS"}})}
showUserButtons(role)}
function showUserButtons(role){var buttonIds=[];switch(role){case "ROLE_ADMIN":case "ROLE_SUPPORT":buttonIds=["admin","user"];break;case "ROLE_EDITOR":buttonIds=["editor"];break;case "ROLE_USER":buttonIds=["user"];break;case "ANONYMOUS":buttonIds=["login"];break;default:buttonIds=["login"]}
$.each(buttonIds,function(index,value){$(".js-btn-"+value).addClass('show')})}
function makeStickyElement(element,offset){offset=(typeof offset!=='undefined')?offset:0;if($(element).length){var elementTop=$(element).offset().top;$(element).addClass('stickyNavbar');if($(element).is(":visible")){$(element).wrap('<div style="height:'+$(element).outerHeight()+'px"></div>')}}
var init=function(){if($(element).length){var scrollTop=$(window).scrollTop();if((scrollTop+offset)>elementTop){$(element).addClass('sticky-active')}else{$(element).removeClass('sticky-active')}}};init();$(window).scroll(function(){init()})}
var fixHeight=(function(){var element='';var _init=function(el,onlyOnBreakpoints){if(typeof onlyOnBreakpoints!=='undefined'){if($.inArray(getResponsiveBreakpoint(),onlyOnBreakpoints)==-1){return}}
element=el;_fixHeight();_onResize()};var _fixHeight=function(){var newHeight=0;$(element).each(function(){$(this).height('auto');if($(this).height()>newHeight){newHeight=$(this).height()}});$(element).each(function(){$(this).height(newHeight)})};var _onResize=function(){$(window).on('resize',function(){_fixHeight()})};return{init:_init}})();var smoothScroll=(function(){var targetHash=null;var _init=function(){setTimeout(function(){$('[href*="#"]:not([href="#"],#tabs a,.tabs a,.scribbrTabs a, #generator a, #citation-generator a)').click(function(e){targetHash=_getHash(this);targetExists=$('#'+targetHash).length;if(targetHash&&targetExists){e.preventDefault();_animateScroll();onScrollToHash('#'+targetHash)}});if(window.location.hash&&window.location.hash.length>1&&/^[a-z0-9]+$/i.test(window.location.hash[1])){if($($(window.location.hash)).length){$('html, body').animate({scrollTop:($(window.location.hash).offset().top-_getOffset())+'px'},1000);onScrollToHash(window.location.hash)}}},300)};var _getHash=function(target){var url=$(target).attr('href');if(typeof url=='undefined'){return!1}
var hash=url.split('#')[1];if(hash){return hash}
return!1};var _animateScroll=function(){var targetPosition=$('#'+targetHash).offset().top;var scrollPosition=targetPosition-_getOffset();$('html, body').animate({scrollTop:scrollPosition},1000)};var _getOffset=function(){var offset=20;$(".stickyNavbar,.buttons.affix-top,.buttons.affix").not('.postTableOfContents').each(function(index){offset+=$(this)[0].clientHeight});offset*=2;$(".sticky-active,.buttons.affix").not('.postTableOfContents').each(function(index){offset-=$(this)[0].clientHeight});if($('.postTableOfContents').length){offset+=50}
return offset};return _init})();function getResponsiveBreakpoint(){var envs=["xs","sm","md","lg","xl"];var env="";var $el=$("<div>");$el.appendTo($("body"));for(var i=envs.length-1;i>=0;i--){env=envs[i];$el.addClass("hidden-"+env+"-up");if($el.is(":hidden")){break}}
$el.remove();return env}
function scribbrTabs(el){$(el).each(function(index){$(this).addClass('nav-tabs');$(this).children('ul').addClass('nav-tabs__nav');$(this).children('ul').children('li').first().addClass('active');$(this).children('ul').nextAll('div').addClass('nav-tabs__tab');$(this).children('ul').next('div').addClass('active');$(this).children('ul').find('a').click(function(event){if(event){event.preventDefault()}
$(this).closest('ul').children('li').removeClass('active');$(this).closest('li').addClass('active');$(this).closest(el).children('div').removeClass('active');$(this).closest(el).children('div'+this.hash).addClass('active')})});if(window.location.hash&&$(window.location.hash).length){$(window.location.hash).closest(el).children('ul').children('li').removeClass('active');$('a[href="'+window.location.hash+'"]').closest('li').addClass('active');$(window.location.hash).closest(el).children('ul').nextAll('div').removeClass('active');$(window.location.hash).addClass('active')}}
function onScrollToHash(hash){if($(hash).hasClass('collapse-header')){$(hash).next().slideDown()}}
function is_touch_device(){return'ontouchstart' in window||navigator.maxTouchPoints}
function kbAnnotatedSample(){if(!is_touch_device()){$(".annotatedPart").on("mouseenter",function(){annotationTriggered("open",this)});$(".annotatedPart").on("mouseleave",function(){annotationTriggered("close",this)});$("#annotationContainer").on("mouseenter",function(){annotationTriggered("open",this)});$("#annotationContainer").on("mouseleave",function(){annotationTriggered("close",this)})}else{$(".annotatedPart").on("click tap",function(){annotationTriggered("open",this)});$("#annotationContainer").on("click tap",function(){annotationTriggered("close",this)})}
$(document).on("click tap",function(e){var container=$("#annotationContainer");var container2=$('.annotatedPart');var target=$(e.target);if(!container.is(target)&&!container2.is(target)&&container.has(target).length===0){container.fadeOut()}});$("a.downloadButton").attr("href",downloadLink)}
var closeAnnotation;function annotationTriggered(event,obj){switch(event){case "open":clearTimeout(closeAnnotation);var annotation=annotations[$(obj).attr("id")];var pos=$(obj).position().top;if($(obj).hasClass("annotatedPart")){pos=pos-10}
if(is_touch_device()&&$(obj).hasClass("annotatedPart")){pos=pos+$(obj).height()+20}
$("#annotationContainer").html(annotation).css("top",pos).fadeIn();break;case "close":closeAnnotation=setTimeout(function(){$("#annotationContainer").fadeOut()},2000);break}}
function triggerAnnontations(event,obj){switch(event){case "open":clearTimeout(closeAnnotation);if($(obj).data('mark-text')){$(obj).addClass('annotatedPart');var annotation=$(obj).data("mark-text");if($(obj).closest('figure').find('#annotationContainer').length===0){$(obj).closest('figure').css('position','relative').prepend('<div id="annotationContainer" />')}}
if($(obj).data('kb-color')){$(obj).closest('figure').find('#annotationContainer').attr('data-border-color',$(obj).data('kb-color'))}
var pos=$(obj).position().top;if($(obj).hasClass("annotatedPart")){pos=pos-10}
if(is_touch_device()&&$(obj).hasClass("annotatedPart")){pos=pos+$(obj).height()+20}
$(obj).closest('figure').find('#annotationContainer').html(annotation).css("top",pos).fadeIn();break;case "close":closeAnnotation=setTimeout(function(){$(obj).find('#annotationContainer').fadeOut()},2000);break}}
$('.kbTooltip').on("mouseenter",function(){triggerAnnontations("open",this)});$('figure.annotatedExample').on("mouseleave",function(){triggerAnnontations("close",this)});$('figure > .expandExample').on('click tap',function(){$(this).closest('figure').css({'max-height':'2000px','overflow-y':'visible'});$(this).remove()});ScribbrGlobal={};ScribbrGlobal.getUser=function(reqValue){try{let token=Cookies.get('token_sfqa');if(window.location.hostname.split('.')[0]==='www'){token=Cookies.get('token')}
if(!token){return null}
let base64Url=token.split('.')[1];let base64=base64Url.replace(/-/g,'+').replace(/_/g,'/');let jsonPayload=decodeURIComponent(atob(base64).split('').map(function(c){return'%'+('00'+c.charCodeAt(0).toString(16)).slice(-2)}).join(''));if(reqValue){return JSON.parse(jsonPayload)[reqValue]}else{return JSON.parse(jsonPayload)}}catch(e){return null}}
function showConsentDialog(){let consentDetails=Cookies.get('CookieConsent');if(typeof consentDetails==="undefined"){return!1}
if(consentDetails){consentDetails=consentDetails.replaceAll(`'`,``);const consentSHowCountries=['ca','gb'];const consentObj=consentDetails.slice(1,consentDetails.length-1).split(',').reduce((consentObj,str)=>{const[key,val]=str.split(':');consentObj[key]=val;return consentObj},{});console.log(consentObj);if(consentSHowCountries.includes(consentObj.region)||Cookiebot.regulations.gdprApplies){Cookiebot.show()}else{alert('Only available for people in the EEA, UK or Canada')}}}
async function getCloudflareJSON(){let data=await fetch('https://www.scribbr.com/cdn-cgi/trace').then(res=>res.text())
let arr=data.trim().split('\n').map(e=>e.split('='))
return Object.fromEntries(arr)}
function kbCheckHeaders(){const freemCountries=['GB','US','DE','BH','CH','BR','LV','PL','FI','RU','IT','IS','DK','AU','IE','RS','AT','TN','FR','MT','KW','CY','CZ','HR','MU','NZ','NL','AL','JP','ES','SE','GR','MA','PT','BS','NO','LT','BE','HU','KZ','SA','AE','CA'];var cfCountry=Cookies.get('cfCountryCode');if(cfCountry===undefined||cfCountry===null||cfCountry===''){getCloudflareJSON().then(function(data){if(typeof data.loc!='undefined'){Cookies.set('cfCountryCode',data.loc,{expires:1})};if(!freemCountries.includes(data.loc)){getServiceText()}})}else{if(!freemCountries.includes(cfCountry)){getServiceText()}}}
function getServiceText(){let serviceItems=document.querySelectorAll(`a[data-service-name="Plagiarism Checker"] h3, a[data-service-name="Plagiarism Checker"] h2, a[data-service-name="Plagiarism Checker"] button, div.service-wrap h2, div.service-wrap h3`);serviceItems.forEach(function(elem){let itemText=elem.innerText.trim();if(elem.textContent.includes('a free check')){elem.innerText=itemText.replace(/a free check/gi,'a plagiarism check')}
if(elem.textContent.includes('a free plagiarism check')){elem.innerText=itemText.replace(/a free plagiarism check/gi,'a plagiarism check')}
if(elem.textContent.includes('Try for free')){elem.innerText=itemText.replace(/Try for free/gi,'Order now')}})}
if(siteLang==='en'&&pageType==='Knowledge Base'){kbCheckHeaders()}