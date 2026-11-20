import os
import tempfile

from django.core import mail
from django.contrib.auth.models import AnonymousUser, User
from django.http import Http404
from django.test import RequestFactory, TestCase, override_settings
from django.urls import reverse

from prediction.models import Prediction
from .views import private_media


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


class PrivateMediaTests(TestCase):
    def test_prediction_image_is_only_served_to_its_owner(self):
        owner = User.objects.create_user(username='image-owner', password='test-password')
        other_user = User.objects.create_user(username='other-user', password='test-password')
        prediction = Prediction.objects.create(
            image='predictions/input/private-image.png',
            predicted_stage='mild',
            confidence=80,
            confidence_scores={'mild': 80},
            created_by=owner,
        )

        with tempfile.TemporaryDirectory() as media_root:
            image_path = os.path.join(media_root, prediction.image.name)
            os.makedirs(os.path.dirname(image_path))
            with open(image_path, 'wb') as image_file:
                image_file.write(b'private image bytes')

            with override_settings(MEDIA_ROOT=media_root):
                request = RequestFactory().get('/media/' + prediction.image.name)
                request.user = owner
                response = private_media(request, prediction.image.name)
                self.assertEqual(b''.join(response.streaming_content), b'private image bytes')
                self.assertEqual(response['Cache-Control'], 'private, no-store')
                response.close()

                request.user = other_user
                with self.assertRaises(Http404):
                    private_media(request, prediction.image.name)

                request.user = AnonymousUser()
                response = private_media(request, prediction.image.name)
                self.assertEqual(response.status_code, 302)
                self.assertIn('/accounts/login/', response['Location'])


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
