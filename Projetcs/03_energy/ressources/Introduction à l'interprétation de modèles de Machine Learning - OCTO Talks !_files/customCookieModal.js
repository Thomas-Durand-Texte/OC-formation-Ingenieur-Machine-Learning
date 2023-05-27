(function($) {
	$( document ).ready(function() {
				/*
		Gestion du scroll sur mobile
		*/
		let isVisible = true;
		let scrollStart = window.scrollY;
		let lastScrollYPosition = window.scrollY;
		let isScrollingDown = true;
		document.addEventListener('scroll', () => {
			if (isScrollingDown && window.scrollY < lastScrollYPosition) {
				isScrollingDown = false
				scrollStart = window.scrollY;
			} else if (!isScrollingDown && window.scrollY > lastScrollYPosition) {
				isScrollingDown = true
				scrollStart = window.scrollY;
			}
			lastScrollYPosition = window.scrollY

			const delta = window.scrollY - scrollStart;
			if (isVisible) {
				if (delta > 500) {
					isVisible = false;
					scrollStart = window.scrollY;
					$('#consentArea #consentAreaModal').slideUp();  
					$('#consentArea #consentAreaButton a').hide(); 
					$('#consentArea #consentAreaButton a').removeClass('active'); 
				}
			}
			else {
				if (delta < -500) {
					isVisible = true;
					scrollStart = window.scrollY;
					$('#consentArea #consentAreaButton a').show(); 
				}
			}
		})

		/*
		Gestion du consentement
		*/
		var toggleConsentDisplay = function() {

			if ($('#consentArea #consentAreaModal').is(':visible')) {

				$('#consentArea #consentAreaModal').slideUp() ;  
				$('#consentArea #consentAreaButton a').removeClass('active') ; 

			}
			else {

				$('#consentArea #consentAreaChoice').hide() ; 
				$('#consentArea #consentAreaIntro').show() ;

				$('#consentArea #consentAreaModal').slideDown() ; 
				$('#consentArea #consentAreaButton a').addClass('active') ; 

			}
		}

		function logConsent(oConsent) {

			var sLogInfo = '🍪 ETAT DES CONSENTEMENTS \n'

			switch (oConsent.google_analytics) {

				case true : {
					sLogInfo += '	- Google Analytics : ✅ Accordé \n' ; 
					break ; 
				}

				case false : {
					sLogInfo += '	- Google Analytics : ❌ Refusé \n' ; 
					break ; 
				}

				default : {
					sLogInfo += '	- Google Analytics : ❓ Inconnu \n' ; 
					break ; 
				}
			}
			switch (oConsent.facebook_pixel) {

				case true : {
					sLogInfo += '	- Facebook Pixel : ✅ Accordé \n' ; 
					break ; 
				}

				case false : {
					sLogInfo += '	- Facebook Pixel : ❌ Refusé\n' ; 
					break ; 
				}

				default : {
					sLogInfo += '	- Facebook Pixel : ❓ Inconnu\n' ; 
					break ; 
				}
			}

			switch (oConsent.hubspot) {

				case true : {
					sLogInfo += '	- Hubspot : ✅ Accordé \n' ; 
					break ; 
				}

				case false : {
					sLogInfo += '	- Hubspot : ❌ Refusé\n' ; 
					break ; 
				}

				default : {
					sLogInfo += '	- Hubspot : ❓ Inconnu\n' ; 
					break ; 
				}
			}

			if (oConsent.when === null) sLogInfo += '	🕐 Dernière modification : Jamais' ; 
			else sLogInfo += '	🕐 Dernière modification : '  + (new Date(oConsent.when).toLocaleString('fr-FR')) ; 

			console.log('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' + sLogInfo + '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n') ; 

		}



		function getConsent() {

			var oCookieConsent = localStorage.getItem('OCAC_cookieConsent') ;

			if (oCookieConsent === null) {

				oCookieConsent = {
					'google_analytics' : null,
					'hubspot' : null,
					'facebook_pixel' : null,
					'when' : null
				} ; 
			}
			else {

				oCookieConsent = JSON.parse(oCookieConsent) ;

				if (!oCookieConsent.hasOwnProperty('google_analytics')) oCookieConsent.google_analytics = null ; 
				if (!oCookieConsent.hasOwnProperty('hubspot')) oCookieConsent.hubspot = null ; 
				if (!oCookieConsent.hasOwnProperty('facebook_pixel')) oCookieConsent.facebook_pixel = null ; 
				if (!oCookieConsent.hasOwnProperty('when')) oCookieConsent.hubspot = null ; 
				else {

					var oTimeMin = new Date() ; 
					oTimeMin.setMonth(oTimeMin.getMonth() - 13) ; 
					oTimeMin = oTimeMin.getTime() ; 

					if (oCookieConsent.when < oTimeMin) {
						oCookieConsent = {
							'google_analytics' : null,
							'hubspot' : null,
							'facebook_pixel' : null,
							'when' : null
						} ; 
					} 
				}
			}

			return oCookieConsent ; 
		}

		function toggleChoices() {

			if ($('#consentArea #consentAreaChoice').is(':visible')) {

				$('#consentArea #consentAreaChoice').fadeOut(200) ;  
				$('#consentArea #consentAreaIntro').delay(200).fadeIn() ; 

			}
			else {
				var oConsent = getConsent() ; 

				$('#consentArea #consentAreaChoice #cookieConsent_ga').attr('checked', oConsent.google_analytics === true) ; 
				$('#consentArea #consentAreaChoice #cookieConsent_hs').attr('checked', oConsent.hubspot === true) ; 
				$('#consentArea #consentAreaChoice #cookieConsent_pixel').attr('checked', oConsent.facebook_pixel === true) ; 
				$('#consentArea #consentAreaIntro').fadeOut(200) ;  
				$('#consentArea #consentAreaChoice').delay(200).fadeIn() ; 
			}
		}


		function updateConsent(bGoogleAnalytics, bHubspot, bPixel) {

			var iToday = new Date().getTime(); 
			var oCookieConsent = {
				'google_analytics' : bGoogleAnalytics,
				'hubspot' : bHubspot,
				'facebook_pixel' : bPixel,
				'when' : iToday
			} ; 

			localStorage.setItem('OCAC_cookieConsent', JSON.stringify(oCookieConsent)) ; 

			logConsent(oCookieConsent) ; 
			activateHubspot(oCookieConsent.hubspot) ; 
			activatePixel(oCookieConsent.facebook_pixel) ; 
			activateGoogleAnalytics(oCookieConsent.google_analytics) ;

		}

		function consentToAll() {
			updateConsent(true, true, true) ; 
			toggleConsentDisplay() ; 
			gtag('event', 'cookie_consent_all', { 'event_category' : 'cookie_consent', 'event_label' : 'Consentement total' }) ;
		}

		function consentNone() {

			updateConsent(false, false, false) ; 
			toggleConsentDisplay() ; 
			gtag('event', 'cookie_consent_none', { 'event_category' : 'cookie_consent', 'event_label' : 'Refuser tout' }) ;
		}

		function consentCustom() {
			updateConsent($('#consentArea #consentAreaChoice #cookieConsent_ga').is(':checked'), $('#consentArea #consentAreaChoice #cookieConsent_hs').is(':checked'), $('#consentArea #consentAreaChoice #cookieConsent_pixel').is(':checked')) ; 
			toggleConsentDisplay() ; 
			gtag('event', 'cookie_consent_custom', { 'event_category' : 'cookie_consent', 'event_label' : 'Consentement personnalisé' }) ;
		}

		function initTrackingAndConsent() {

			oConsent = getConsent() ;
			logConsent(oConsent) ;

			if (oConsent.hubspot !== null) activateHubspot(oConsent.hubspot) ;
			if (oConsent.facebook_pixel !== null) activatePixel(oConsent.facebook_pixel) ;
			activateGoogleAnalytics(oConsent.google_analytics) ; 

			if (oConsent.google_analytics === null || oConsent.hubspot === null || oConsent.facebook_pixel === null) {
				toggleConsentDisplay() ; 
				gtag('event', 'cookie_consent_unknown', { 'event_category' : 'cookie_consent', 'event_label' : 'Consentement inconnu' }) ;
			}

		}

		function activatePixel(bActivate) {
			if (bActivate === true) {
				if (typeof gtag !== 'function') {
					// Cas ou Google Analytics n'était pas encore chargé
					window.dataLayer = window.dataLayer || [];
					window.gtag = function() {dataLayer.push(arguments);}
					gtag('js', new Date());
				}
				gtag('event', 'axeptio_activate_facebook_pixel')
				// END ajout

				console.log('🟢 Activation du tracking Facebook Pixel') ; 
				if (typeof fbq === 'function')
					fbq('consent', 'grant');
			}
			else {
				if (typeof fbq === 'function')
					fbq('consent', 'revoke');
			}
		}
		function activateHubspot(bActivate) {

			var _hsp = window._hsp = window._hsp || [];

			if (bActivate === true) {

				// Ajout
				if (typeof gtag !== 'function') {

					// Cas ou Google Analytics n'était pas encore chargé
					window.dataLayer = window.dataLayer || [];
					window.gtag = function() {dataLayer.push(arguments);}

					gtag('js', new Date());
				}
				gtag('event', 'axeptio_activate_hubspot')
				// END ajout

				console.log('🟢 Activation du tracking Hubspot') ; 
				_hsp.push(['doNotTrack', {track: true}]) ; 
			}
			else {

				if ($('script#hs-script-loader').length) {

					console.log('🔴 Désactivation du tracking Hubspot') ; 
					_hsp.push(['revokeCookieConsent']) ; 
				}
			}
		}

		function activateGoogleAnalytics(bActivate) {

			if (bActivate === true) {
				var sAnalyticsStorage = 'granted' ;
				var sLog = '🟢 Activation du tracking Google Analytics, avec cookies' ;
			}
			else {
				var sAnalyticsStorage = 'denied' ;
				var sLog = '🟠 Activation du tracking Google Analytics, sans cookies' ;
			}

			sLog += ' [analytics_storage : ' + sAnalyticsStorage + ']' ;

			console.log(sLog) ; 
			if (typeof gtag !== 'function') {

				// Cas ou Google Analytics n'était pas encore chargé
				window.dataLayer = window.dataLayer || [];
				window.gtag = function() {dataLayer.push(arguments);}
/*
				if (bActivate === true) {
					gtag('config', 'UA-3350117-35', {'anonymise_ip': true});
					gtag('consent', 'UA-3350117-35', {
							'ad_storage': 'denied',
							'analytics_storage': sAnalyticsStorage
					});
					gtag('js', new Date());
				}
*/
			}
			if (bActivate === true) {					
				// Ajout
				gtag('event', 'axeptio_activate_google_analytics')
				// END ajout
			}
		}

		$('#consentArea #consentAreaButton a').click(function (event) {
			toggleConsentDisplay() ; 
			event.preventDefault() ; 
		}) ;

		$('#consentArea #consentAreaIntro a.yes, #consentArea #consentAreaChoice a.accept-all').click(function (event) {
			consentToAll() ; 
			event.preventDefault() ; 
		}) ;

		$('#consentArea #consentAreaIntro a.no').click(function (event) {
			consentNone() ; 
			event.preventDefault() ; 
		}) ;
		$('#consentArea #consentAreaIntro a.choose, #consentArea #consentAreaChoice a.back').click(function (event) {
			toggleChoices() ; 
			event.preventDefault() ; 
		}) ;
		$('#consentArea #consentAreaChoice a.finish').click(function (event) {
			consentCustom() ; 
			event.preventDefault() ; 
		}) ;

		initTrackingAndConsent() ; 
	}) ;
})(jQuery)
