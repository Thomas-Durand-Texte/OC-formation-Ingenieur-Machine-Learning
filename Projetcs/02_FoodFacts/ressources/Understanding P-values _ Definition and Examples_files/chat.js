ScribbrChat={};setTimeout(function(){initMsgBird()},3000);ScribbrChat.getCookie=function(name){let value="; "+document.cookie;let parts=value.split("; "+name+"=");if(parts.length===2){return parts.pop().split(";").shift()}}
ScribbrChat.getUser=function(){try{let token=ScribbrChat.getCookie('token_'+window.location.hostname.split('.')[0]);if(window.location.hostname.split('.')[1]!=='scribbr'||window.location.hostname.split('.')[0]==='order'){token=ScribbrChat.getCookie('token')}
if(!token){return null}
let base64Url=token.split('.')[1];let base64=base64Url.replace(/-/g,'+').replace(/_/g,'/');let jsonPayload=decodeURIComponent(atob(base64).split('').map(function(c){return'%'+('00'+c.charCodeAt(0).toString(16)).slice(-2)}).join(''));return JSON.parse(jsonPayload)}catch(e){return null}}
ScribbrChat.isKb=function(){if(typeof knowledgeBase==='undefined'){return!1}
return knowledgeBase}
ScribbrChat.isCitationGenerator=function(){return typeof(pageType)!=='undefined'&&pageType==='Citation Generator'}
ScribbrChat.show=function(){if(ScribbrChat.isKb()||ScribbrChat.prevPageIsKb()||ScribbrChat.isCitationGenerator()){window.MessageBirdChatWidget.hide()}else{window.MessageBirdChatWidget.init()}}
ScribbrChat.initialize=function(){if(ScribbrChat.isKb()||ScribbrChat.prevPageIsKb()||ScribbrChat.isCitationGenerator()){return!1}else{return!0}}
ScribbrChat.setUpAttributes=function(){let attributes={domain:window.location.hostname,}
let user=ScribbrChat.getUser();if(!user){return window.MessageBirdChatWidget.setAttributes(attributes)}
if(user.name){user.first_name=user.name.first;user.last_name=user.name.last;if(user.name.infix){user.last_name=`${user.name.infix} ${user.name.last}`}}
if(user.roles&&user.roles[0]){user.role=user.roles[0]}
attributes.user_id=`${user.id}`;attributes.first_name=`${user.first_name}`;attributes.last_name=`${user.last_name}`;if(user.phoneNumber){attributes.phone=`${user.phoneNumber}`}
attributes.email=`${user.username}`;attributes.user_role=`${user.role}`;return window.MessageBirdChatWidget.setAttributes(attributes)}
ScribbrChat.addEventListeners=function(){document.addEventListener('click',function(event){if(event.target.getAttribute('data-toggle')==='open-chat'&&window.MessageBirdChatWidget.isOpen===!1){window.MessageBirdChatWidget.toggleChat()}},!1)}
function initMsgBird(){if(!ScribbrChat.initialize())
return;window.MessageBirdChatWidgetSettings={widgetId:'f1969c26-6920-470c-b7be-d37a782c2acc',initializeOnLoad:!0,};!function(){"use strict";if(Boolean(document.getElementById("live-chat-widget-script")))console.error("MessageBirdChatWidget: Snippet loaded twice on page");else{var e,t;window.MessageBirdChatWidget={},window.MessageBirdChatWidget.queue=[];for(var i=["init","setConfig","toggleChat","identify","hide","on","shutdown"],n=function(){var e=i[d];window.MessageBirdChatWidget[e]=function(){for(var t=arguments.length,i=new Array(t),n=0;n<t;n++)i[n]=arguments[n];window.MessageBirdChatWidget.queue.push([[e,i]])}},d=0;d<i.length;d++)n();var a=(null===(e=window)||void 0===e||null===(t=e.MessageBirdChatWidgetSettings)||void 0===t?void 0:t.widgetId)||"",o=function(){var e,t=document.createElement("script");t.type="text/javascript",t.src="https://livechat.messagebird.com/bootstrap.js?widgetId=".concat(a),t.async=!0,t.id="live-chat-widget-script";var i=document.getElementsByTagName("script")[0];null==i||null===(e=i.parentNode)||void 0===e||e.insertBefore(t,i)};"complete"===document.readyState?o():window.attachEvent?window.attachEvent("onload",o):window.addEventListener("load",o,!1)}}();window.MessageBirdChatWidget.on('ready',function(){ScribbrChat.show();ScribbrChat.setUpAttributes();ScribbrChat.addEventListeners()})}(()=>{if(typeof knowledgeBase==='undefined'||typeof Cookies==='undefined'){return}
var cookie=Cookies.get('prev_page_is_kb');if(typeof cookie==='undefined'){Cookies.set('prev_page_is_kb',[knowledgeBase,!1]);return}
cookie=JSON.parse(cookie);cookie[1]=cookie[0];cookie[0]=knowledgeBase;Cookies.set('prev_page_is_kb',cookie)})()
ScribbrChat.prevPageIsKb=function(){if(typeof Cookies==='undefined'){return}
var cookie=Cookies.get('prev_page_is_kb');if(typeof cookie==='undefined'){return!1}
cookie=JSON.parse(cookie);return cookie[1]}
if(typeof getLanguage==='undefined'){const getLanguage=function(){return typeof siteLang==='undefined'?getSfLanguage():siteLang}}
if(typeof getSfLanguage==='undefined'){const getSfLanguage=function(){return document.getElementsByTagName("html")[0].getAttribute("lang")}}
if(typeof hide_show_contact_chat_buttons==='undefined'){const hide_show_contact_chat_buttons=function(){}}