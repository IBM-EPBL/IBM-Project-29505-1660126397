(window.__LOADABLE_LOADED_CHUNKS__=window.__LOADABLE_LOADED_CHUNKS__||[]).push([[10],{"++iY":function(e,t,r){"use strict";r.d(t,"e",(function(){return a})),r.d(t,"a",(function(){return c})),r.d(t,"g",(function(){return s})),r.d(t,"k",(function(){return d})),r.d(t,"j",(function(){return _})),r.d(t,"b",(function(){return l})),r.d(t,"l",(function(){return E})),r.d(t,"d",(function(){return p})),r.d(t,"m",(function(){return f})),r.d(t,"f",(function(){return u})),r.d(t,"h",(function(){return I})),r.d(t,"o",(function(){return R})),r.d(t,"p",(function(){return b})),r.d(t,"n",(function(){return T})),r.d(t,"q",(function(){return N})),r.d(t,"i",(function(){return O})),r.d(t,"c",(function(){return g}));var o=r("RHI1"),n=r.n(o);const i=["objectType","actionType","productTitle","category","productVersion","processType"];function a(e,t){return m("Deleted Object",{objectType:"Comment",projectId:e,notebookId:t})}function c(e,t){return m("Created Object",{objectType:"Comment",projectId:e,notebookId:t})}function s(e,t){return m("Updated Object",{objectType:"Comment",projectId:e,notebookId:t})}function d(e){const{scope:t,contentFilter:r}=e;return m("Exported Object",{objectType:"Notebook",target:"Github",scope:t,contentFilter:r})}function _(e){const{projectId:t,notebookId:r,contentType:o,language:n,accessType:i,runtimeType:a}=e;return m("Created Object",{objectType:"Code snippet (ITC)",projectId:t,notebookId:r,contentType:o,language:n,accessType:i,runtimeType:a})}function l(e){const{projectId:t,notebookId:r,contentType:o}=e;return m("Created Object",{objectType:"Asset",projectId:t,notebookId:r,contentType:o})}function E(e){const{projectId:t,notebookId:r,contentType:o}=e;return m("Updated Object",{objectType:"Asset",projectId:t,notebookId:r,contentType:o})}function p(e,t){return m("Created Object",{objectType:"Notebook version",projectId:e,notebookId:t})}function f(e,t){return m("Updated Object",{objectType:"Notebook version",projectId:e,notebookId:t})}function u(e,t){return m("Deleted Object",{objectType:"Notebook version",projectId:e,notebookId:t})}function I(e,t){return m("Updated Object",{objectType:"Notebook",projectId:e,notebookId:t})}function R(e){const{projectId:t,notebookId:r,customized:o}=e;return m("Started Process",{objectType:"Runtime",processType:"Started",projectId:t,notebookId:r,customized:o})}function b(e,t){return m("Started Process",{objectType:"Runtime",processType:"Stopped",projectId:e,notebookId:t})}function T(e,t){return m("Started Process",{objectType:"Runtime",processType:"Restarted",projectId:e,notebookId:t})}function N(e,t){return m("Read Object",{objectType:"Notebook",projectId:e,notebookId:t})}function O(e){const{projectId:t,notebookId:r,errorCode:o}=e;return m("Read Object",{objectType:"Error Page",projectId:t,notebookId:r,errorCode:o})}function g(e){const{projectId:t,notebookId:r,notebookLanguage:o,type:n,runtimeType:i,cpu:a,ram:c,gpu:s,gpuType:d,baseSoftwareSpecId:_,softwarePackage:l,remoteLocation:E,remoteLocationId:p}=e;return m("Created Object",{objectType:"Notebook",projectId:t,notebookId:r,notebookLanguage:o,type:n,runtimeType:i,cpu:a,ram:c,gpu:s,gpuType:d,baseSoftwareSpecId:_,softwarePackage:l,remoteLocation:E,remoteLocationId:p})}function m(e,t){var r;if(null!==t&&"object"==typeof t&&(t.productTitle="Watson Studio",t.category="Notebook"),!window)return void console.log("No window object found, could not send tracking events");let o=window,a=null;for(;o;){var c;if(o.analytics){a=o.analytics;break}o=null!==(c=o.parent)&&void 0!==c&&c.window&&o!==o.parent.window?o.parent.window:null}null!==(r=a)&&void 0!==r&&r.track&&a.track(e,function(e){return n()(e,(e,t)=>i.includes(t)?t:"custom."+t)}(t))}},LhCx:function(e,t,r){"use strict";r.r(t);var o=r("OLAl"),n=r("q1tI"),i=r.n(n);var a=r("tQch");var c=r("mwIZ"),s=r.n(c);var d=r("OlNQ"),_={uiBackground:"#f4f4f4",interactive01:"#0f62fe",interactive02:"#393939",interactive03:"#0f62fe",interactive04:"#0f62fe",danger:"#da1e28",ui01:"#fff",ui02:"#f4f4f4",ui03:"#e0e0e0",ui04:"#8d8d8d",ui05:"#161616",text01:"#161616",text02:"#525252",text03:"#a8a8a8",text04:"#fff",text05:"#6f6f6f",link01:"#0f62fe",inverseLink:"#78a9ff",icon01:"#161616",icon02:"#525252",icon03:"#fff",field01:"#fff",field02:"#f4f4f4",inverse01:"#fff",inverse02:"#393939",support01:"#da1e28",support02:"#198038",support03:"#f1c21b",support04:"#0043ce",inverseSupport01:"#fa4d56",inverseSupport02:"#42be65",inverseSupport03:"#f1c21b",inverseSupport04:"#4589ff",overlay01:"rgba(22,22,22,.5)",focus:"#0f62fe",inverseFocusUi:"#fff",hoverPrimary:"#0353e9",hoverPrimaryText:"#0043ce",hoverSecondary:"#4c4c4c",hoverTertiary:"#0353e9",hoverUi:"#e5e5e5",hoverSelectedUi:"#cacaca",hoverDanger:"#b81921",hoverRow:"#e5e5e5",inverseHoverUi:"#4c4c4c",activePrimary:"#002d9c",activeSecondary:"#6f6f6f",activeTertiary:"#002d9c",activeUi:"#c6c6c6",activeDanger:"#750e13",selectedUi:"#e0e0e0",highlight:"#d0e2ff",skeleton01:"#e5e5e5",skeleton02:"#c6c6c6",visitedLink:"#8a3ffc",disabled01:"#fff",disabled02:"#c6c6c6",disabled03:"#8d8d8d",brand01:"#0f62fe",brand02:"#393939",brand03:"#0f62fe",selected:"#e0e0e0",dapNavHeight:"48px",dapNavActionbarHeight:"88px",errorPage:"ErrorPage__errorPage___23ZFJ",topWrapper:"ErrorPage__topWrapper___DRbQr",errorTitle:"ErrorPage__errorTitle___8YMCy",description:"ErrorPage__description___LIQ2T",errorDetailsBox:"ErrorPage__errorDetailsBox___b1PgZ",errorDetailsTitle:"ErrorPage__errorDetailsTitle___YI5nN",errorDetailsContent:"ErrorPage__errorDetailsContent___SMgIH",enabled:"ErrorPage__enabled___3r35K"};var l=r("gQMU"),E=r.n(l);function p(e,t,r){const{formatMessage:o}=e;let n;switch(t){case a.h:n=r.isRemoved?function(e,t){const{formatMessage:r}=e;let o="",n="";return Array.isArray(t.supportedKernels)&&t.supportedKernels.length>0&&(o=E()(s()(t.supportedKernels[0],"language","")),n=s()(t.supportedKernels[0],"display_name",o)),[i.a.createElement("p",{key:"line1"},r({id:"ERROR_PAGE_REMOVED_RUNTIME_DEFINITION_LINE1"},{kernelLanguageVersion:n})),i.a.createElement("p",{key:"line2"},i.a.createElement("span",null,r({id:"ERROR_PAGE_REMOVED_RUNTIME_DEFINITION_ADVICE"},{kernelLanguage:o})))]}(e,r):r.isExpected?function(e,t){const{formatMessage:r}=e;let o="";return Array.isArray(t.supportedKernels)&&t.supportedKernels.length>0&&(o=s()(t.supportedKernels[0],"display_name",s()(t.supportedKernels[0],"language",""))),[i.a.createElement("p",{key:"line1"},r({id:"ERROR_PAGE_UNINSTALLED_RUNTIME_DEFINITION_LINE1"},{kernelLanguageVersion:o})),i.a.createElement("p",{key:"line2"},i.a.createElement("span",null,r({id:"ERROR_PAGE_UNINSTALLED_RUNTIME_DEFINITION_ADVICE"})))]}(e,r):function(e,t){const{formatMessage:r}=e,o=t.runtimeDefinitionId?t.runtimeDefinitionId:"";return[i.a.createElement("p",{key:"line1"},r({id:"ERROR_PAGE_RUNTIME_DEFINITION_MISSING_LINE1"},{runtimeDefinitionId:o})),i.a.createElement("p",{key:"line2"},i.a.createElement("span",null,r({id:"ERROR_PAGE_RUNTIME_DEFINITION_MISSING_ADVICE"})))]}(e,r);break;case a.f:n=function(e,t){const{formatMessage:r}=e;let o="...";return Array.isArray(t.componentNames)&&(o=t.componentNames.sort().join(", ")),[i.a.createElement("p",{key:"line1"},r({id:"ERROR_PAGE_PROXY_RUNTIME_DEFINITION_MISSING_LINE1"},{componentNames:o})),i.a.createElement("p",{key:"line2"},i.a.createElement("span",null,r({id:"ERROR_PAGE_PROXY_RUNTIME_DEFINITION_MISSING_ADVICE"})))]}(e,r);break;case a.i:n=function(e,t,r){const{formatMessage:o}=e,n=s()(r,"owner"),a=s()(r,"plan");return[i.a.createElement("p",{key:"line1"},o({id:"ERROR_PAGE_USAGE_LIMIT_REACHED_LINE1"},{plan_name:a})),i.a.createElement("p",{key:"line2"},o({id:"ERROR_PAGE_USAGE_LIMIT_REACHED_LINE2"},{project_owner:n})),i.a.createElement("p",{key:"line3"},o({id:"ERROR_PAGE_USAGE_LIMIT_REACHED_LINE3"},{link_tag:e=>i.a.createElement(d.a,{target:"_blank",href:"/settings/apps?context=cpdaas&referral=upgrade",className:_.link},e)})),i.a.createElement("p",{key:"line4"},o({id:"ERROR_PAGE_USAGE_LIMIT_REACHED_LINE4"},{link_tag:e=>i.a.createElement(d.a,{target:"_blank",href:"/docs/content/getting-started/plans.html"},e)}))]}(e,0,r);break;case a.c:case a.g:case a.a:n=function(e,t){const{formatMessage:r}=e,o=[i.a.createElement("p",{key:"line1"},r({id:"ERROR_PAGE_NO_CONTAINER_RESOURCES_LINE1"})),i.a.createElement("p",{key:"line2"},r({id:"ERROR_PAGE_NO_CONTAINER_RESOURCES_LINE2"}))];return null!=t&&t.errorMessage&&o.push(i.a.createElement("p",{key:"line3"},t.errorMessage)),o}(e,r);break;case a.n:n=function(e){const{formatMessage:t}=e;return[i.a.createElement("p",{key:"line1"},t({id:"ERROR_PAGE_NO_VERSION_AVAILABLE_LINE1"})),i.a.createElement("p",{key:"line2"},t({id:"ERROR_PAGE_NO_VERSION_AVAILABLE_LINE2"})),i.a.createElement("p",{key:"line3"},t({id:"ERROR_PAGE_NO_VERSION_AVAILABLE_LINE3"})),i.a.createElement("p",{key:"line4"},t({id:"ERROR_PAGE_NO_VERSION_AVAILABLE_LINE4"})),i.a.createElement("p",{key:"line5"},t({id:"ERROR_PAGE_NO_VERSION_AVAILABLE_LINE5"}))]}(e);break;case a.m:n=o({id:"ERROR_PAGE_NOTEBOOK_LINK_BROKEN_LINE1"});break;case a.b:n=r.message;break;case a.l:n=function(e,t,r){const{formatMessage:o}=e,n=s()(r,"activeEnvironmentName","Unknown");return[i.a.createElement("p",{key:"line1"},o({id:"COMMON_MULTIPLE_INTERACTIVE_RUNTIMES_IN_PROJECT_SUBTITLE_2"},{active_runtime_name:n,tool_name:o({id:"COMMON_TOOL_JUPYTERLAB"})})),i.a.createElement("p",{key:"line2"},i.a.createElement("span",null,o({id:"COMMON_MULTIPLE_INTERACTIVE_RUNTIMES_IN_PROJECT_ACTION_2"},{active_runtime_name:n,tool_name:o({id:"COMMON_TOOL_JUPYTERLAB"})})))]}(e,0,r);break;case a.j:case a.k:n=function(e,t){const{formatMessage:r}=e;let o=[];return o=t===a.k?[i.a.createElement("p",{key:"line1"},r({id:"ERROR_PAGE_NO_GIT_REPO_LINE1"}))]:[i.a.createElement("p",{key:"line1"},r({id:"ERROR_PAGE_GIT_TOKEN_MISSING_LINE1"}))],o}(e,t);break;case a.e:default:n=null}return n}var f=r("17x9"),u=r.n(f),I=r("JZM8"),R=r.n(I),b=r("TSYQ"),T=r.n(b),N=r("a1iI"),O={uiBackground:"#161616",interactive01:"#0f62fe",interactive02:"#6f6f6f",interactive03:"#fff",interactive04:"#4589ff",danger:"#da1e28",ui01:"#262626",ui02:"#393939",ui03:"#393939",ui04:"#6f6f6f",ui05:"#f4f4f4",text01:"#f4f4f4",text02:"#c6c6c6",text03:"#6f6f6f",text04:"#fff",text05:"#8d8d8d",link01:"#78a9ff",inverseLink:"#0f62fe",icon01:"#f4f4f4",icon02:"#c6c6c6",icon03:"#fff",field01:"#262626",field02:"#393939",inverse01:"#161616",inverse02:"#f4f4f4",support01:"#fa4d56",support02:"#42be65",support03:"#f1c21b",support04:"#4589ff",inverseSupport01:"#da1e28",inverseSupport02:"#24a148",inverseSupport03:"#f1c21b",inverseSupport04:"#0f62fe",overlay01:"rgba(0,0,0,.65)",focus:"#fff",inverseFocusUi:"#0f62fe",hoverPrimary:"#0353e9",hoverPrimaryText:"#a6c8ff",hoverSecondary:"#606060",hoverTertiary:"#f4f4f4",hoverUi:"#353535",hoverSelectedUi:"#4c4c4c",hoverDanger:"#b81921",hoverRow:"#353535",inverseHoverUi:"#e5e5e5",activePrimary:"#002d9c",activeSecondary:"#393939",activeTertiary:"#c6c6c6",activeUi:"#525252",activeDanger:"#750e13",selectedUi:"#393939",highlight:"#002d9c",skeleton01:"#353535",skeleton02:"#525252",visitedLink:"#be95ff",disabled01:"#262626",disabled02:"#525252",disabled03:"#8d8d8d",brand01:"#0f62fe",brand02:"#6f6f6f",brand03:"#fff",selected:"#393939",dapNavHeight:"48px",dapNavActionbarHeight:"88px",errorPage:"ErrorPage-dark__errorPage___a7Sxc",topWrapper:"ErrorPage-dark__topWrapper___Ct2Ny",errorTitle:"ErrorPage-dark__errorTitle___g6H0W",description:"ErrorPage-dark__description___rsRht",errorDetailsBox:"ErrorPage-dark__errorDetailsBox___t4bO3",errorDetailsTitle:"ErrorPage-dark__errorDetailsTitle___yGIQ+",errorDetailsContent:"ErrorPage-dark__errorDetailsContent___XIMFE",enabled:"ErrorPage-dark__enabled___YoixS"},g={uiBackground:"#f4f4f4",interactive01:"#0f62fe",interactive02:"#393939",interactive03:"#0f62fe",interactive04:"#0f62fe",danger:"#da1e28",ui01:"#fff",ui02:"#f4f4f4",ui03:"#e0e0e0",ui04:"#8d8d8d",ui05:"#161616",text01:"#161616",text02:"#525252",text03:"#a8a8a8",text04:"#fff",text05:"#6f6f6f",link01:"#0f62fe",inverseLink:"#78a9ff",icon01:"#161616",icon02:"#525252",icon03:"#fff",field01:"#fff",field02:"#f4f4f4",inverse01:"#fff",inverse02:"#393939",support01:"#da1e28",support02:"#198038",support03:"#f1c21b",support04:"#0043ce",inverseSupport01:"#fa4d56",inverseSupport02:"#42be65",inverseSupport03:"#f1c21b",inverseSupport04:"#4589ff",overlay01:"rgba(22,22,22,.5)",focus:"#0f62fe",inverseFocusUi:"#fff",hoverPrimary:"#0353e9",hoverPrimaryText:"#0043ce",hoverSecondary:"#4c4c4c",hoverTertiary:"#0353e9",hoverUi:"#e5e5e5",hoverSelectedUi:"#cacaca",hoverDanger:"#b81921",hoverRow:"#e5e5e5",inverseHoverUi:"#4c4c4c",activePrimary:"#002d9c",activeSecondary:"#6f6f6f",activeTertiary:"#002d9c",activeUi:"#c6c6c6",activeDanger:"#750e13",selectedUi:"#e0e0e0",highlight:"#d0e2ff",skeleton01:"#e5e5e5",skeleton02:"#c6c6c6",visitedLink:"#8a3ffc",disabled01:"#fff",disabled02:"#c6c6c6",disabled03:"#8d8d8d",brand01:"#0f62fe",brand02:"#393939",brand03:"#0f62fe",selected:"#e0e0e0",dapNavHeight:"48px",dapNavActionbarHeight:"88px",errorPage:"ErrorPage-light__errorPage___ge3YA",topWrapper:"ErrorPage-light__topWrapper___KB8E6",errorTitle:"ErrorPage-light__errorTitle___mOjJx",description:"ErrorPage-light__description___thm7b",errorDetailsBox:"ErrorPage-light__errorDetailsBox___ids7b",errorDetailsTitle:"ErrorPage-light__errorDetailsTitle___XsPbH",errorDetailsContent:"ErrorPage-light__errorDetailsContent___2UvlC",enabled:"ErrorPage-light__enabled___084vk"},m=r("++iY");let h=_;class v extends i.a.Component{constructor(e,t){super(e,t);const r=s()(e,"error.code");this.state={code:r,isHelpVisible:!1},h=Object(N.getStyles)(t.themeSettings,_,g,O)}componentDidMount(){this.props.user&&(Object(o.d)(this.props.intl,this.props.context,this.props.projectId,s()(this.props,"project.resource.entity.name",""),s()(this.props,"notebookAsset.resource.metadata.name","")),Object(o.a)(),Object(m.i)({projectId:this.props.projectId,notebookId:this.props.notebookId,errorCode:this.state.code}))}_onToggleHelp(){this.setState({isHelpVisible:!this.state.isHelpVisible})}_renderErrorDetailsBox(e){const{formatMessage:t}=this.props.intl;let r;const o=s()(this.props,"error.trace");if(this.state.code||o){let n,a,c;o&&(n=i.a.createElement("p",null,"Log ID: ",i.a.createElement("span",{id:"requestIDContainer"},o))),this.state.code&&(a=i.a.createElement("p",null,t({id:"ERROR_PAGE_CODE"},{code:this.state.code})));const s=e?R()(e,["projectId","notebookId","environmentId","environmentDisplayName","runtimeDefinitionId","supportedKernels"]):{};s&&"object"==typeof s&&Object.keys(s).length>0&&(c=i.a.createElement("pre",null,JSON.stringify(s,null,2)));const d=T()({[h.errorDetailsBox]:!0,[h.enabled]:this.state.isHelpVisible});r=i.a.createElement("div",{className:d},i.a.createElement("div",{className:h.errorDetailsTitle,onClick:this._onToggleHelp.bind(this)},t({id:"ERROR_PAGE_MORE_HELP"})),i.a.createElement("div",{className:h.errorDetailsContent},i.a.createElement("p",null,t({id:"ERROR_PAGE_GET_IN_TOUCH"})),n,a,c))}return r}render(){const e=s()(this.props,"error.params",{}),t=function(e,t){const{formatMessage:r}=e;let o;switch(t){case a.f:case a.h:o=r({id:"ERROR_PAGE_RUNTIME_DEFINITION_MISSING_TITLE"});break;case a.i:o=r({id:"ERROR_PAGE_USAGE_LIMIT_REACHED_TITLE"});break;case a.e:o=r({id:"ERROR_PAGE_NOTEBOOK_NOT_SHARED_TITLE"});break;case a.n:o=r({id:"ERROR_PAGE_NO_VERSION_AVAILABLE_TITLE"});break;case a.m:o=r({id:"ERROR_PAGE_NOTEBOOK_LINK_BROKEN_TITLE"});break;case a.c:case a.g:case a.a:o=r({id:"ERROR_PAGE_NO_CONTAINER_RESOURCES_TITLE"});break;case a.b:o=r({id:"ERROR_PAGE_ENVIRONMENT_DEPRECATED_TITLE"});break;case a.l:o=r({id:"ERROR_PAGE_MULTIPLE_JUPYTERLAB_RUNTIMES_TITLE"});break;case a.j:o=r({id:"ERROR_PAGE_GIT_TOKEN_MISSING_TITLE"});break;case a.k:o=r({id:"ERROR_PAGE_NO_GIT_REPO_TITLE"});break;default:o=r({id:"ERROR_PAGE_DEFAULT_ERROR_TITLE"})}return o}(this.props.intl,this.state.code),r=p(this.props.intl,this.state.code,e),o=this._renderErrorDetailsBox(e);return i.a.createElement("div",{className:h.errorPage},i.a.createElement("div",{className:h.topWrapper},i.a.createElement("h1",{className:h.errorTitle},t)),i.a.createElement("div",{className:h.description},r),o)}}v.contextTypes={themeSettings:u.a.object,isFeatureEnabled:u.a.func.isRequired,deploymentTarget:u.a.string.isRequired},v.propTypes={intl:u.a.object.isRequired,projectId:u.a.string,notebookId:u.a.string,context:u.a.string,user:u.a.object,notebookAsset:u.a.object,project:u.a.object};var k=r("ANjH"),A=r("/MKj"),y=r("wSuE"),P=r("2OET"),L=r("Ty5D");t.default=Object(k.compose)(P.c,y.hot,L.o,Object(A.connect)((function(e){return{notebookAsset:e.notebookAsset,project:e.project,error:e.error}}),null))(v)},UzQL:function(e,t,r){"use strict";r.d(t,"a",(function(){return N})),r.d(t,"b",(function(){return O}));var o=r("LhCx"),n=r("17x9"),i=r.n(n),a=r("q1tI"),c=r.n(a),s=r("i8i4"),d=r.n(s),_=r("mwIZ"),l=r.n(_);const E=()=>{let e=null,t=window;try{for(;t;){if((r=t.globalHeader)&&"function"==typeof r.showNotification){e=t.globalHeader;break}t=t!==t.parent.window?t.parent.window:null}}catch(e){console.log("Could not find global header which might be due to cross origin setup")}var r;return e};var p=r("2OET"),f=r("V/vL");class u extends c.a.Component{constructor(e){super(e),this.state={errorMode:!1,visibleModal:!1,modalContent:null}}getChildContext(){return{isFeatureEnabled:this._isFeatureEnabled.bind(this),openModal:this._toggleModal.bind(this,!0),closeModal:this._toggleModal.bind(this,!1),deploymentTarget:this.props.deploymentTarget,themeSettings:this.props.themeSettings,showErrorPage:this._showErrorPage.bind(this),showNotification:this._showNotification.bind(this),hideNotification:this._hideNotification.bind(this)}}_showErrorPage(e){this.setState({errorMode:e})}_isFeatureEnabled(e){return Boolean(l()(this.props,"user.features."+e))}_toggleModal(e,t){this.setState({visibleModal:e||!this.state.visibleModal,modalContent:t})}_onClose(){this.setState({visibleModal:null,modalContent:null})}_showNotification(e){const t=E();return t?Promise.all([I(e.title),I(e.subtitle),I(e.caption)]).then(r=>{const o="error"===e.type?0:10,n=Number.isInteger(e.dismissal)?e.dismissal:o;return t.showNotification({...e,dismissal:n,title:r[0],subtitle:r[1],caption:r[2]})}):Promise.resolve()}_hideNotification(e){const t=E();t&&t.hideNotification(e)}render(){if(this.state.errorMode)return c.a.createElement("div",{id:"notebookWrapper"},c.a.createElement(o.default,{projectId:this.props.projectId,notebookId:this.props.notebookId,context:this.props.context,user:this.props.user}));let e=null;return this.state.modalContent&&this.state.visibleModal&&(e=c.a.cloneElement(this.state.modalContent,{visible:!0,onClose:this._onClose.bind(this)})),c.a.createElement("div",{id:"notebookWrapper"},Object(f.renderRoutes)(this.props.route.routes,this.props),e)}}function I(e){return e?"string"==typeof e?Promise.resolve(e):new Promise(t=>{const r=window.document.createElement("div");d.a.render(e,r,()=>{t(r.innerHTML)})}):Promise.resolve()}const R=Object(p.c)(u);u.childContextTypes={isFeatureEnabled:i.a.func.isRequired,openModal:i.a.func,closeModal:i.a.func,deploymentTarget:i.a.string,themeSettings:i.a.object,showErrorPage:i.a.func,showNotification:i.a.func.isRequired,hideNotification:i.a.func.isRequired},u.propTypes={themeSettings:i.a.object,intl:i.a.object.isRequired,spaId:i.a.string.isRequired,projectId:i.a.string,notebookId:i.a.string,context:i.a.string,user:i.a.object,route:i.a.object,deploymentTarget:i.a.string};var b=R,T=r("P1eC");const N=(e,t,r)=>{const o={locale:e,default_locale:t,message_bundles:[...r]};return Object(T.initializeIntl)(o)},O=(e,t)=>{const{...r}=e;return c.a.createElement(p.b,{value:t},c.a.createElement(b,r))}},tQch:function(e,t,r){"use strict";r.d(t,"f",(function(){return o})),r.d(t,"h",(function(){return n})),r.d(t,"i",(function(){return i})),r.d(t,"e",(function(){return a})),r.d(t,"n",(function(){return c})),r.d(t,"m",(function(){return s})),r.d(t,"c",(function(){return d})),r.d(t,"b",(function(){return _})),r.d(t,"g",(function(){return l})),r.d(t,"a",(function(){return E})),r.d(t,"d",(function(){return p})),r.d(t,"o",(function(){return f})),r.d(t,"l",(function(){return u})),r.d(t,"j",(function(){return I})),r.d(t,"k",(function(){return R}));const o="ERROR_PROXY_RUNTIME_DEFINITION_MISSING",n="ERROR_RUNTIME_DEFINITION_MISSING",i="ERROR_USAGE_LIMIT_REACHED",a="ERROR_NOTEBOOK_NOT_SHARED",c="NOTEBOOK_VERSION_NOT_AVAILABLE",s="NOTEBOOK_LINK_BROKEN",d="ERROR_GPU_RESOURCES_UNAVAILABLE",_="ERROR_ENVIRONMENT_DEPRECATED",l="ERROR_RESOURCES_UNAVAILABLE",E="ERROR_CONTAINER_RESOURCES_UNAVAILABLE",p="ERROR_MISSING_REQUIRED_PROJECT_TOKEN",f="NOT_YET_IMPLEMENTED",u="JUPYTER_LAB_RUNTIME_ACTIVE",I="JUPYTER_LAB_GIT_TOKEN_MISSING",R="JUPYTER_LAB_PROJECT_NOT_CONNECTED_TO_REPO"}}]);