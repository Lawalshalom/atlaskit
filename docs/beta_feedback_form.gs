/**
 * Creates the AtlasKit beta-tester feedback form and a linked responses spreadsheet.
 *
 * 1. Open https://script.new while signed in to Google.
 * 2. Replace the editor's contents with this file and click Run.
 * 3. Approve the permissions (it is your own script). The form, edit and responses
 *    links appear in the Execution log.
 */
function createAtlasKitBetaForm() {
  const PLAYGROUND = 'https://heou1vjm7ockft-8080.proxy.runpod.net';
  const REPO = 'https://github.com/Lawalshalom/atlaskit';

  const form = FormApp.create('AtlasKit beta feedback');
  form.setDescription(
    'Thank you for testing AtlasKit, the open-source developer toolkit for N-ATLaS, Nigeria\'s ' +
    'Yoruba, Hausa, Igbo and Nigerian English language model. This form takes about 5 minutes. ' +
    'Your answers are part of the validation evidence for our National AI Innovation Challenge 2026 submission.\n\n' +
    'Playground: ' + PLAYGROUND + '\nCode and docs: ' + REPO);
  form.setConfirmationMessage('Thank you! Your feedback helps make N-ATLaS easier for every Nigerian developer to build on.');
  form.setAllowResponseEdits(true);
  form.setShowLinkToRespondAgain(false);
  form.setAcceptingResponses(true);
  // Google Workspace forms can be limited to your organisation; testers are external.
  try { form.setRequireLogin(false); } catch (e) { /* not a Workspace account */ }

  const isUrl = FormApp.createTextValidation().requireTextIsUrl()
      .setHelpText('Please paste a full link, starting with https://').build();
  const isEmail = FormApp.createTextValidation().requireTextIsEmail().build();

  form.addSectionHeaderItem().setTitle('About you');
  form.addTextItem().setTitle('Full name').setRequired(true);
  form.addTextItem().setTitle('Role and company or organisation')
      .setHelpText('For example: Backend developer, Acme Fintech').setRequired(true);
  form.addTextItem().setTitle('GitHub or LinkedIn profile URL')
      .setHelpText('This lets the judges confirm you are a real, external developer.')
      .setValidation(isUrl).setRequired(true);
  form.addTextItem().setTitle('Email (optional)')
      .setHelpText('Only if you are happy for us to follow up.').setValidation(isEmail);
  form.addTextItem().setTitle('Tester ID from your invite (for example T3)');
  form.addMultipleChoiceItem().setTitle('Are you a member or employee of Halo Digital, the team behind AtlasKit?')
      .setChoiceValues(['No', 'Yes']).setRequired(true);

  form.addSectionHeaderItem().setTitle('What you tried');
  form.addCheckboxItem().setTitle('Which parts of AtlasKit did you use?')
      .setChoiceValues(['Playground (browser)', 'Python SDK', 'JavaScript SDK',
                        'OpenAI SDK pointed at AtlasKit (drop-in)', 'atlaskit CLI', 'curl or raw HTTP'])
      .showOtherOption(true).setRequired(true);
  form.addMultipleChoiceItem()
      .setTitle('How long did it take from opening the README or playground to your first successful N-ATLaS response?')
      .setChoiceValues(['Under 5 minutes', '5 to 15 minutes', '15 to 30 minutes', 'More than 30 minutes',
                        'I did not get a response'])
      .setRequired(true);
  form.addCheckboxItem().setTitle('Which languages did you try?')
      .setChoiceValues(['Yoruba', 'Hausa', 'Igbo', 'Nigerian English']).setRequired(true);
  form.addGridItem().setTitle('For the languages you speak, how good were N-ATLaS\'s answers?')
      .setHelpText('1 = poor, 5 = excellent. Choose "Did not try" for the others.')
      .setRows(['Yoruba', 'Hausa', 'Igbo', 'Nigerian English'])
      .setColumns(['1', '2', '3', '4', '5', 'Did not try']);
  form.addMultipleChoiceItem().setTitle('Did you try speech (transcription or voice chat)?')
      .setChoiceValues(['Yes, and it worked well', 'Yes, it worked but with mistakes', 'Yes, but it failed', 'No'])
      .setRequired(true);
  form.addScaleItem()
      .setTitle('How easy was AtlasKit to use, compared with setting up and calling the model yourself?')
      .setBounds(1, 5).setLabels('Much harder', 'Much easier').setRequired(true);

  form.addSectionHeaderItem().setTitle('Your feedback');
  form.addParagraphTextItem().setTitle('What broke or confused you?');
  form.addParagraphTextItem().setTitle('What would you build with AtlasKit and N-ATLaS?');
  form.addMultipleChoiceItem().setTitle('Would you use AtlasKit in a real project?')
      .setChoiceValues(['Yes', 'Maybe', 'No']).setRequired(true);
  form.addMultipleChoiceItem().setTitle('May we quote your feedback in our NAIC 2026 submission?')
      .setChoiceValues(['Yes, with my name', 'Yes, but anonymously', 'No']).setRequired(true);

  const sheet = SpreadsheetApp.create('AtlasKit beta feedback (responses)');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, sheet.getId());
  // Newer Google Forms must be published before anyone can respond.
  try { form.setPublished(true); } catch (e) { /* older Forms: always published */ }

  let link = form.getPublishedUrl();
  try { link = form.shortenFormUrl(link); } catch (e) { /* keep the long link */ }
  Logger.log('SHARE WITH TESTERS: ' + link);
  Logger.log('Edit the form:      ' + form.getEditUrl());
  Logger.log('Responses sheet:    ' + sheet.getUrl());
}
