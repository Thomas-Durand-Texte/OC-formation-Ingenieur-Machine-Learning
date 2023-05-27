//main navigation click events
let navItems = document.querySelectorAll(`.main-nav .nav__list a`);
navItems.forEach(function (elem) {
  elem.addEventListener("click", function () {    
    var mainItem,
        subItem,
        itemURL;
    //main item or subnav
    if (elem.classList.contains('sub-item')) {
      subItem = elem.innerText;
      mainItem = getParentCategory(elem.closest('li.nav-dropdown'));
      itemURL = elem.getAttribute('href');
    } else {
      subItem = '';
      mainItem = getParentCategory(elem.closest('li.nav-dropdown'));
      itemURL = elem.getAttribute('href');
    }
    var eventProperties = {
      "main_item": mainItem,
      "sub_item": subItem,
      "destination_url": itemURL,
      "location_url": ampliCurrentURL
    };
    amplitude.getInstance().logEvent('Header Clicked', eventProperties);
    
  });
});
let uploadButtons = document.querySelectorAll(`a[href*="${rootUrl}/plagiarism-checker"], a[href*="${rootUrl}/turnitin-check"], a[href*="${rootUrl}/order"]`);
uploadButtons.forEach(function (elem) {
  elem.addEventListener("click", function () {
    let uploadUrl = elem.getAttribute('href').split('?')[0],
        buttonService = '';
    if(uploadUrl.includes(`${rootUrl}/plagiarism-checker`)) {
      buttonService = 'plagiarism checker';
    } else if (uploadUrl.includes(`${rootUrl}/order`)) {
      buttonService = 'proofreading & editing';
    }
    let buttonText = elem.innerText,
        buttonType = getLocationByClass(elem);
    var eventProperties = {
      "location_url": ampliCurrentURL,
      'destination_url': uploadUrl,
      "button_text": buttonText,
      "service": buttonService,
      "button_type": buttonType,
    };
    amplitude.getInstance().logEvent('Upload Button Clicked', eventProperties);
  });
});
//get parent details
function getParentCategory(parentEl) {
  var cls = parentEl.classList,
      parentCategory;
  switch (true) {
    case cls.contains('proofreading'):
      parentCategory = 'proofreading';
      break;
    case cls.contains('plagiarismcheck'):
      parentCategory = 'plagiarism';
      break;
    case cls.contains('citation'):
      parentCategory = 'citation tool';
      break;
    default:
      parentCategory = 'knowledgebase';
  }
  return parentCategory;
} 
//get upload button location by class
function getLocationByClass(buttonElement) {
  var cls = buttonElement.classList,
      buttonLocation;
  switch (true) {
    case cls.contains('btn--upload'):
    case cls.contains('dropdown__link'):
      buttonLocation = 'header';
      break;
    case cls.contains('btn--primary'):
      buttonLocation = 'primary button';
      break;
    case cls.contains('btn--secondary'):
        buttonLocation = 'secondary button';
        break;
    default:
      buttonLocation = 'other';
  }
  return buttonLocation;
}

//subnav clicked
let subNavItems = document.querySelectorAll(`.sub-nav .nav__list a`);
subNavItems.forEach(function (elem) {
  elem.addEventListener("click", function () {
    //main item
    var mainItem = elem.innerText,
    itemURL = elem.getAttribute('href');
    var eventProperties = {
      "main_item": mainItem,
      "destination_url": itemURL,
      "location_url": ampliCurrentURL
    };
    amplitude.getInstance().logEvent('Sub Header Clicked', eventProperties);

  });
});
//if knowledgebase target service and document download
if (document.body.classList.contains('knowledgebase')) {
  window.addEventListener('load', (event) => {
    let servicesLinks = document.querySelectorAll(`.cta-services a, .scw-location-in_text a, .scw-location-exit_intent a`);
    servicesLinks.forEach(function (elem) {
      elem.addEventListener("click", function () {
        var linkType,
          itemService;
        if (elem.dataset.serviceName) {
          itemService = elem.dataset.serviceName;
        }
        if (elem.dataset.serviceLocation) {
          linkType = elem.dataset.serviceLocation;
        } else if (elem.closest('div.cta-services').classList.contains('style-vertical')){
          linkType = 'sidebar';
        } else if (elem.closest('div.cta-services').classList.contains('style-horizontal')) {
          linkType = 'header';
        } else if (elem.closest('div.scw').classList.contains('scw-location-exit_intent')) {
          linkType = 'exit_intent';
        } else {
          linkType = 'in_text';
        }
        var linkURL = elem.getAttribute('href').split('?')[0],
            serviceCopy = '';
        if (elem.getElementsByTagName('H3')[0]) {
            serviceCopy = elem.getElementsByTagName('H3')[0].innerText;
        }
        var eventProperties = {
          "ad_type": linkType,
          "location_url": ampliCurrentURL,
          "outgoing_url": linkURL,
          "service": itemService,
          "ad_copy": serviceCopy
        };
        amplitude.getInstance().logEvent('Ad Clicked Knowledge Base', eventProperties);
      });
    });
  });
  //track document download
  let documentLinks = document.querySelectorAll(`article a[href$=".pptx"], article a[href$=".docx"], article a[href$=".pdf"], article a[href$=".xlxs"]`);
  documentLinks.forEach(function(elem){
    elem.addEventListener('click', function(){
      let fileURL = elem.getAttribute('href');
          documentType = fileURL.split('.').pop(),
          documentName = fileURL.substring(fileURL.lastIndexOf('/') + 1);
      
      let eventProperties = {
        "document_type": documentType,
        "location_url": ampliCurrentURL,
        "document_name": documentName,
        "document_url": fileURL,
        "category": current_category
      };

      amplitude.getInstance().logEvent('Document Downloaded', eventProperties);
    })
    
  });
}
//homepage click tracking
if (document.body.classList.contains('page-template-home-twig')) {
  let servicesLinks = document.querySelectorAll(`.products-box-wrapper a`);
  servicesLinks.forEach(function (elem) {
    elem.addEventListener("click", function () {
      var itemService;
      if (elem.dataset.serviceName) {
        itemService = elem.dataset.serviceName;
      }
      let linkURL = new URL(elem.getAttribute('href').split('?')[0]);
      var serviceCopy = elem.innerText;
      var eventProperties = {
        "destination_url": linkURL.href,
        "destination_path": linkURL.pathname,
        "button_text": serviceCopy,
        "service": itemService,
      };
      amplitude.getInstance().logEvent('Home Page Service Clicked', eventProperties);
    });
  });
}
//track citation free checker opened
window.addEventListener('load', (event) => {
  if (typeof bindButtons === 'function') {
    let citChkButtons = document.querySelectorAll(`a.btn--white, a.ts-cc`);
    citChkButtons.forEach(function(elem){
      elem.addEventListener('click', function(){
        amplitude.getInstance().logEvent('Citation Free Checker Opened');
      })  
    });
  }
});
//track gender free checker click
if(siteLang == 'de') {
  let genderFreeButtons = document.querySelectorAll(`a[href*="scribbr.de/gendern/genderpruefung"]`);
  genderFreeButtons.forEach(function (elem) {
    elem.addEventListener("click", function () {
      amplitude.getInstance().logEvent('Gender Free Checker Opened');
    });
  });
}
//navigation upload button clicked
let navOrangeButtons = document.querySelectorAll(`.btn--upload`);
navOrangeButtons.forEach(function (elem) {
  elem.addEventListener("click", function () {
    let linkURL = new URL(elem.getAttribute('href').split('?')[0], ampliCurrentURL);
    var buttonText = elem.innerText;
    var eventProperties = {
      "location_url": ampliCurrentURL,
      "location_path": window.location.pathname,
      "destination_url": linkURL.href,
      "destination_path": linkURL.pathname,
      "button_text": buttonText,
    };
    amplitude.getInstance().logEvent('Header Button Clicked', eventProperties);
  });
});

//onclick simple page events
function sendAmpliEvent($eventName) {
  amplitude.getInstance().logEvent($eventName);
}