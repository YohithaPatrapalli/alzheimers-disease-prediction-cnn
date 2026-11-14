from django.core import mail
from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse


class PublicInformationPageTests(TestCase):
    def test_about_page_renders(self):
        response = self.client.get(reverse('core:about'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'About the Prediction System')
        self.assertContains(response, 'not a medical diagnosis')

    def test_contact_page_renders_and_sends_valid_submission(self):
        response = self.client.get(reverse('core:contact'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Send Message')

        with override_settings(
            EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
            CONTACT_EMAIL='support@example.com',
        ):
            response = self.client.post(
                reverse('core:contact'),
                {
                    'name': 'Test User',
                    'email': 'test@example.com',
                    'subject': 'Page feedback',
                    'message': 'The contact page is working.',
                },
                follow=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Message sent successfully')
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, ['support@example.com'])
        self.assertIn('The contact page is working.', mail.outbox[0].body)


class LogoutTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='logout-user', password='test-password')
        self.client.force_login(self.user)

    def test_logout_link_submits_post_and_ends_the_session(self):
        response = self.client.get(reverse('core:about'))

        self.assertContains(response, reverse('core:logout'))
        self.assertContains(response, 'csrfmiddlewaretoken')

        response = self.client.post(reverse('core:logout'))

        self.assertRedirects(response, reverse('home'))
        self.assertNotIn('_auth_user_id', self.client.session)
